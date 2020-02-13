package tpami.basealgorithmlearning.datagathering;
import java.io.File;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.api4.java.common.reconstruction.IReconstructible;
import org.api4.java.datastructure.kvstore.IKVStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import ai.libs.jaicore.basic.reconstruction.ReconstructionPlan;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.sql.rest.IRestDatabaseConfig;
import ai.libs.jaicore.db.sql.rest.RestSqlAdapter;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.timing.TimedComputation;
import weka.classifiers.AbstractClassifier;

public class BaseLearnerExperimenter {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	private static IExperimentDatabaseHandle databaseHandle;
	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout to = new Timeout(1, TimeUnit.HOURS);

	public static void main(final String[] args) throws Exception {

		/* get experiment configuration */
		final Class<?> classifierClass = Class.forName(args[0]);
		final IBaseLearnerExperimentConfig config = ConfigFactory.create(IBaseLearnerExperimentConfig.class);
		String classifierWorkingName = classifierClass.getSimpleName().toLowerCase();
		config.loadPropertiesFromFile(new File("conf/experiments/defaultparams/baselearner.conf"));

		/* setup database connection */
		IRestDatabaseConfig dbConfig = ConfigFactory.create(IRestDatabaseConfig.class);
		dbConfig.loadPropertiesFromFile(new File("conf/dbcon-rest.conf"));
		final RestSqlAdapter adapter = new RestSqlAdapter(dbConfig);
		databaseHandle = new ExperimenterMySQLHandle(adapter, "evaluations_classifiers_" + classifierWorkingName);

		//		/* prepare database */
		//		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(config, databaseHandle);
		//		preparer.setLoggerName("example");
		//		preparer.synchronizeExperiments();
		//		System.exit(0);

		/* run an experiment */
		final ObjectMapper om = new ObjectMapper();
		System.out.println("Creating the runner.");

		AtomicLong memoryPeak = new AtomicLong();

		Thread t = new Thread() {
			@Override
			public void run() {
				while (!Thread.interrupted()) {
					long currentMemoryConsumption = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
					memoryPeak.set(Math.max(memoryPeak.get(), currentMemoryConsumption));
					System.out.println("Current consumption " + currentMemoryConsumption + ", max: " + memoryPeak.get());
					try {
						Thread.sleep(1000);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		};
		t.start();

		ExperimentRunner runner = new ExperimentRunner(config, new IExperimentSetEvaluator() {

			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException {
				System.out.println("Reading in experiment.");
				Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
				int seed = Integer.parseInt(keys.get("seed"));
				int openmlid = Integer.parseInt(keys.get("openmlid"));
				int datapoints = Integer.parseInt(keys.get("datapoints"));

				System.out.println("Running " + classifierClass.getName() + " on dataset " + openmlid + " with seed " + seed + " and " + datapoints + " data points.");
				Map<String, Object> map = new HashMap<>();

				/* load dataset */
				List<ILabeledDataset<?>> splitTmp = null;
				IWekaClassifier cTmp = null;
				Optional<IKVStore> rowInBoundTable = null;
				try {
					Dataset ds = (Dataset)OpenMLDatasetReader.deserializeDataset(openmlid);
					if (ds.getLabelAttribute() instanceof INumericAttribute) {
						System.out.println("Converting numeric dataset to classification dataset!");
						ds = (Dataset)DatasetUtil.convertToClassificationDataset(ds);
					}

					/* check whether the dataset is reproducible */
					if (ds.getConstructionPlan().getInstructions().isEmpty()) {
						throw new IllegalStateException("Construction plan for dataset is empty!");
					}

					/* check whether we have reports that even smaller sizes do not work */
					rowInBoundTable = adapter.getRowsOfTable("executionbounds_" + classifierWorkingName).stream().filter(r -> r.getAsInt("openmlid") == openmlid).findAny();
					System.out.println("Bound available: " + (rowInBoundTable.isPresent() ? "yes (" + rowInBoundTable.get().getAsInt("datapoints") + ")" : "no"));
					if (rowInBoundTable.isPresent() && rowInBoundTable.get().getAsInt("datapoints") < datapoints) {
						throw new IllegalStateException("Experiment canceled due to lower bound on other experiments on this dataset.");
					}

					System.out.println("Label: " + ds.getLabelAttribute().getClass().getName() + " ... " + ds.getLabelAttribute().getStringDescriptionOfDomain());

					if (datapoints > ds.size()) {
						logError(rowInBoundTable, openmlid, datapoints, adapter, classifierWorkingName);
						throw new IllegalStateException("Invalid Experiment, added or modified entry in database if relevant.");
					}
					double portion = datapoints * 1.0 / ds.size();
					splitTmp = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, portion);

					/* get json describing training and test data */
					JsonNode trainNode = om.valueToTree(((IReconstructible)splitTmp.get(0)).getConstructionPlan());
					JsonNode testNode = om.valueToTree(((IReconstructible)splitTmp.get(1)).getConstructionPlan());
					ArrayNode trainInstructions = (ArrayNode)trainNode.get("instructions");
					ArrayNode testInstructions = (ArrayNode)testNode.get("instructions");
					ArrayNode commonPrefix = om.createArrayNode();
					int numInstructions = trainInstructions.size();
					System.out.println("Train instructions consist of " + numInstructions + " instructions.");
					for (int i = 0; i < numInstructions - 1; i++) {
						if (!trainInstructions.get(i).equals(testInstructions.get(i))) {
							throw new IllegalStateException("Train and test data do not have common prefix!\nTrain: " + trainInstructions.get(i) + "\nTest: " + testInstructions.get(i));
						}
						commonPrefix.add(trainInstructions.get(i));
					}
					if (commonPrefix.size() == 0) {
						throw new IllegalStateException("The common prefix of train and test data must not be empty.\nTrain instructions:\n" + trainInstructions + "\nTest instructions:\n" + testInstructions);
					}

					/* create classifier */
					//					ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(config, databaseHandle);
					//					preparer.setLoggerName("example");
					//					preparer.synchronizeExperiments();
					//					Syst
					cTmp = new WekaClassifier(AbstractClassifier.forName(classifierClass.getName(), null));

					/* now write core data about this experiment */
					map.put("evaluationinputdata", commonPrefix.toString());
					if (map.get("evaluationinputdata").equals("[]")) {
						throw new IllegalStateException();
					}
					map.put("traindata", trainInstructions.get(numInstructions - 1).toString());
					map.put("testdata", testInstructions.get(numInstructions - 1).toString());
					map.put("pipeline", om.writeValueAsString(((IReconstructible)cTmp).getConstructionPlan()));

					/* verify that the compose data recover to the same */
					ObjectNode composed = om.createObjectNode();
					composed.set("instructions", om.readTree(map.get("evaluationinputdata").toString()));
					ArrayNode an = (ArrayNode)composed.get("instructions");
					JsonNode trainDI = om.readTree(map.get("traindata").toString());
					JsonNode testDI = om.readTree(map.get("testdata").toString());
					an.add(trainDI);
					if (!om.readValue(composed.toString(), ReconstructionPlan.class).reconstructObject().equals(splitTmp.get(0))) {
						throw new IllegalStateException("Reconstruction of train data failed!");
					}
					an.remove(an.size() - 1);
					an.add(testDI);
					if (!om.readValue(composed.toString(), ReconstructionPlan.class).reconstructObject().equals(splitTmp.get(1))) {
						throw new IllegalStateException("Reconstruction of test data failed!");
					}

					System.out.println(map);
					processor.processResults(map);
					map.clear();
				}
				catch (Throwable e) {
					throw new ExperimentEvaluationFailedException(e);
				}
				final List<ILabeledDataset<?>> split = splitTmp;
				final IWekaClassifier c = cTmp;

				/* now train classifier */
				SimpleDateFormat format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
				map.put("train_start",  format.format(new Date(System.currentTimeMillis())));
				try {
					memoryPeak.set(0);
					TimedComputation.compute(() -> { c.fit(split.get(0)); return null;}, to.milliseconds(), "Experiment timeout exceeded.");
					Thread.sleep(1000);
				} catch (Throwable e) {
					map.put("train_end",  format.format(new Date(System.currentTimeMillis())));
					processor.processResults(map);
					try {
						logError(rowInBoundTable, openmlid, datapoints, adapter, classifierWorkingName);
					} catch (SQLException e1) {
						e1.printStackTrace();
					}
					throw new ExperimentEvaluationFailedException(e);
				}
				map.put("train_end",  format.format(new Date(System.currentTimeMillis())));
				map.put("memory_peak", memoryPeak.get());
				System.out.println("Finished training, now testing. Memory peak was " + map.get("memory_peak"));
				map.put("test_start",  format.format(new Date(System.currentTimeMillis())));
				List<Integer> gt = new ArrayList<>();
				List<Integer> pr = new ArrayList<>();
				try {
					int n = split.get(1).size();
					for (ILabeledInstance i : split.get(1)) {
						gt.add((int)i.getLabel());
						pr.add((int)c.predict(i).getPrediction());
						System.out.println(gt.size() + "/" + n + " (" + (gt.size() * 100.0 / n) + "%)");
					}
				}
				catch (Throwable e) {
					map.put("test_end",  format.format(new Date(System.currentTimeMillis())));
					processor.processResults(map);
					throw new ExperimentEvaluationFailedException(e);
				}
				map.put("test_end", format.format(new Date(System.currentTimeMillis())));
				map.put("gt", gt);
				map.put("pr", pr);
				processor.processResults(map);
				System.out.println("Finished Experiment " + experimentEntry.getExperiment().getValuesOfKeyFields() + ". Results: " + map);
			}
		}, databaseHandle);

		System.out.println("Running random experiments with timeout " + TOTAL_RUNTIME_IN_SECONDS + "s.");
		runner.setLoggerName("example");
		long start = System.currentTimeMillis();
		while ((System.currentTimeMillis() - start) / 1000 <= TOTAL_RUNTIME_IN_SECONDS - to.seconds() * 1.1) {
			System.out.println("Conducting next experiment.");
			try {
				runner.randomlyConductExperiments(1);
			}
			catch (Throwable e) {
				e.printStackTrace();
			}
		}
		System.out.println("No more time left to conduct more experiments. Stopping.");
		System.exit(0);
	}

	public static void logError(final Optional<IKVStore> rowInBoundTable, final int openmlid, final int datapoints, final IDatabaseAdapter adapter, final String classifierWorkingName) throws SQLException {
		Map<String, Object> errorMap = new HashMap<>();
		errorMap.put("openmlid", openmlid);
		errorMap.put("datapoints", datapoints);
		if (!rowInBoundTable.isPresent()) {
			adapter.insert("executionbounds_" + classifierWorkingName, errorMap);
		}
		else if (datapoints < rowInBoundTable.get().getAsInt("datapoints")) {
			Map<String, Object> condMap = new HashMap<>();
			condMap.put("openmlid", openmlid);
			adapter.update("executionbounds_" + classifierWorkingName, errorMap, condMap);
		}
	}
}
