package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.reconstruction.IReconstructible;
import org.api4.java.datastructure.kvstore.IKVStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;

import ai.libs.jaicore.basic.reconstruction.ReconstructionPlan;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.timing.TimedComputation;
import tpami.basealgorithmlearning.datagathering.PeakMemoryObserver;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.rules.OneR;

public class DefaultMetaLearnerExperimenter {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout to = new Timeout(1, TimeUnit.HOURS);

	public static void main(final String[] args) throws Exception {

		// /* prepare database */
		// ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(config, databaseHandle);
		// preparer.setLoggerName("example");
		// preparer.synchronizeExperiments();
		// System.exit(0);

		DefaultMetaLearnerConfigContainer container = new DefaultMetaLearnerConfigContainer(args[0], args[1]);
		Class<?> classifierClass = Class.forName(args[1]);
		String classifierWorkingName = classifierClass.getSimpleName().toLowerCase();
		IExperimentDatabaseHandle databaseHandle = container.getDatabaseHandle();

		/* run an experiment */
		final ObjectMapper om = new ObjectMapper();

		final PeakMemoryObserver mobs = new PeakMemoryObserver();
		mobs.start();

		EventBus eventBus = new EventBus("metalearners");

		LOGGER.info("Creating the runner.");
		ExperimentRunner runner = new ExperimentRunner(container.getConfig(), new IExperimentSetEvaluator() {

			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException {
				LOGGER.info("Reading in experiment.");
				Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
				int seed = Integer.parseInt(keys.get("seed"));
				int openmlid = Integer.parseInt(keys.get("openmlid"));
				int datapoints = Integer.parseInt(keys.get("datapoints"));

				LOGGER.info("Running {} on dataset {} with seed {} and {} data points.", classifierClass.getName(), openmlid, seed, datapoints);
				Map<String, Object> map = new HashMap<>();

				/* load dataset */
				List<ILabeledDataset<?>> splitTmp = null;
				IWekaClassifier cTmp = null;
				Optional<IKVStore> rowInBoundTable = null;
				try {
					Dataset ds = (Dataset) OpenMLDatasetReader.deserializeDataset(openmlid);
					if (ds.getLabelAttribute() instanceof INumericAttribute) {
						LOGGER.info("Converting numeric dataset to classification dataset!");
						ds = (Dataset) DatasetUtil.convertToClassificationDataset(ds);
					}

					/* check whether the dataset is reproducible */
					if (ds.getConstructionPlan().getInstructions().isEmpty()) {
						throw new IllegalStateException("Construction plan for dataset is empty!");
					}

					/* check whether we have reports that even smaller sizes do not work */
					rowInBoundTable = container.getAdapter().getRowsOfTable("executionbounds_" + classifierWorkingName).stream().filter(r -> r.getAsInt("openmlid") == openmlid).findAny();
					LOGGER.info("Bound available: {}", (rowInBoundTable.isPresent() ? "yes (" + rowInBoundTable.get().getAsInt("datapoints") + ")" : "no"));
					if (rowInBoundTable.isPresent() && rowInBoundTable.get().getAsInt("datapoints") < datapoints) {
						throw new IllegalStateException("Experiment canceled due to lower bound on other experiments on this dataset.");
					}

					LOGGER.info("Label: {} ... {}", ds.getLabelAttribute().getClass().getName(), ds.getLabelAttribute().getStringDescriptionOfDomain());
					if (datapoints > ds.size()) {
						logError(rowInBoundTable, openmlid, datapoints, container.getAdapter(), classifierWorkingName);
						throw new IllegalStateException("Invalid Experiment, added or modified entry in database if relevant.");
					}
					double portion = datapoints * 1.0 / ds.size();
					splitTmp = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, portion);

					/* get json describing training and test data */
					JsonNode trainNode = om.valueToTree(((IReconstructible) splitTmp.get(0)).getConstructionPlan());
					JsonNode testNode = om.valueToTree(((IReconstructible) splitTmp.get(1)).getConstructionPlan());
					ArrayNode trainInstructions = (ArrayNode) trainNode.get("instructions");
					ArrayNode testInstructions = (ArrayNode) testNode.get("instructions");
					ArrayNode commonPrefix = om.createArrayNode();
					int numInstructions = trainInstructions.size();
					LOGGER.info("Train instructions consist of {} instructions.", numInstructions);
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
					// ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(config, databaseHandle);
					// preparer.setLoggerName("example");
					// preparer.synchronizeExperiments();
					// Syst
					cTmp = new WekaClassifier(AbstractClassifier.forName(classifierClass.getName(), null));

					AbstractClassifier baseLearnerToUse = new OneR();

					if (cTmp.getClassifier() instanceof SingleClassifierEnhancer) {
						((SingleClassifierEnhancer) cTmp.getClassifier()).setClassifier(new LeakingBaselearnerWrapper(eventBus, baseLearnerToUse));
					} else if (cTmp.getClassifier() instanceof MultipleClassifiersCombiner) {
						((MultipleClassifiersCombiner) cTmp.getClassifier()).setClassifiers(new AbstractClassifier[] { new LeakingBaselearnerWrapper(eventBus, baseLearnerToUse) });
					} else {
						throw new RuntimeException("Unknown super class of meta learner: " + cTmp.getClassifier().getClass());
					}

					/* now write core data about this experiment */
					map.put("evaluationinputdata", commonPrefix.toString());
					if (map.get("evaluationinputdata").equals("[]")) {
						throw new IllegalStateException();
					}
					map.put("traindata", trainInstructions.get(numInstructions - 1).toString());
					map.put("testdata", testInstructions.get(numInstructions - 1).toString());
					map.put("pipeline", om.writeValueAsString(((IReconstructible) cTmp).getConstructionPlan()));

					/* verify that the compose data recover to the same */
					ObjectNode composed = om.createObjectNode();
					composed.set("instructions", om.readTree(map.get("evaluationinputdata").toString()));
					ArrayNode an = (ArrayNode) composed.get("instructions");
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

					LOGGER.info("{}", map);
					processor.processResults(map);
					map.clear();
				} catch (Throwable e) {
					throw new ExperimentEvaluationFailedException(e);
				}
				final List<ILabeledDataset<?>> split = splitTmp;
				final IWekaClassifier c = cTmp;

				/* now train classifier */
				SimpleDateFormat format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
				map.put("train_start", format.format(new Date(System.currentTimeMillis())));
				long deadlinetimestamp = System.currentTimeMillis() + to.milliseconds();
				try {
					mobs.reset();
					TimedComputation.compute(() -> {
						c.fit(split.get(0));
						return null;
					}, to.milliseconds(), "Experiment timeout exceeded.");
					Thread.sleep(1000);
				} catch (Throwable e) {
					map.put("train_end", format.format(new Date(System.currentTimeMillis())));
					processor.processResults(map);
					if (e instanceof AlgorithmTimeoutedException) {
						try {
							logError(rowInBoundTable, openmlid, datapoints, container.getAdapter(), classifierWorkingName);
						} catch (SQLException e1) {
							e1.printStackTrace();
						}
					}
					throw new ExperimentEvaluationFailedException(e);
				}
				map.put("train_end", format.format(new Date(System.currentTimeMillis())));
				map.put("memory_peak", mobs.getMaxMemoryConsumptionObserved());
				LOGGER.info("Finished training, now testing. Memory peak was {}", map.get("memory_peak"));
				map.put("test_start", format.format(new Date(System.currentTimeMillis())));
				List<Integer> gt = new ArrayList<>();
				List<Integer> pr = new ArrayList<>();
				try {
					long lastTimeoutCheck = 0;
					int n = split.get(1).size();
					for (ILabeledInstance i : split.get(1)) {
						if (System.currentTimeMillis() - lastTimeoutCheck > 10000) {
							lastTimeoutCheck = System.currentTimeMillis();
							long remainingTime = deadlinetimestamp - lastTimeoutCheck;
							LOGGER.debug("Remaining time for this classifier: {}", remainingTime);
							if (remainingTime <= 0) {
								LOGGER.info("Triggering timeout ...");
								throw new AlgorithmTimeoutedException(0);
							}
						}
						gt.add((int) i.getLabel());
						pr.add((int) c.predict(i).getPrediction());
						LOGGER.info("{}/{} ({}%)", gt.size(), n, gt.size() * 100.0 / n);
					}
				} catch (Throwable e) {
					map.put("test_end", format.format(new Date(System.currentTimeMillis())));
					processor.processResults(map);
					throw new ExperimentEvaluationFailedException(e);
				}
				map.put("test_end", format.format(new Date(System.currentTimeMillis())));
				map.put("gt", gt);
				map.put("pr", pr);
				processor.processResults(map);
				LOGGER.info("Finished Experiment {}. Results: {}", experimentEntry.getExperiment().getValuesOfKeyFields(), map);
			}

			@Subscribe
			public void parseLeakingBaselearnerEvent(LeakingBaselearnerEvent event) {
				System.out.println("Received event: " + event.getTimestamp() + " - " + event.getEventType());
			}
		}, databaseHandle);

		// TODO
		eventBus.register(runner);

		LOGGER.info("Runner created.");

		LOGGER.info("Running random experiments with timeout " + TOTAL_RUNTIME_IN_SECONDS + "s.");
		runner.setLoggerName("example");
		long start = System.currentTimeMillis();
		while ((System.currentTimeMillis() - start) / 1000 <= TOTAL_RUNTIME_IN_SECONDS - to.seconds() * 1.1) {
			LOGGER.info("Conducting next experiment.");
			try {
				runner.randomlyConductExperiments(1);
			} catch (Throwable e) {
				e.printStackTrace();
			}
		}
		LOGGER.info("No more time left to conduct more experiments. Stopping.");
		System.exit(0);
	}

	/**
	 * We use execution bounds ONLY for the cases that it is completely safe to avoid execution.
	 * This can only happen in two cases:
	 * 1. the TRAINING already times out for a smaller subset. We can, in general, assume that train time grows monotonically in the train size.
	 * It would be not sufficient to put a bound if the OVERALL or TEST runtime exceeds the time bound.
	 * This is because the test time will reduce if the train data will be increased.
	 *
	 * 2. if the train data in the experiment exceeds the actually available data in the dataset.
	 *
	 * Both together imply that an experiment can be avoided if its dataset specification is greate than one that already failed for one of the two reasons above.
	 *
	 */
	public static void logError(final Optional<IKVStore> rowInBoundTable, final int openmlid, final int datapoints, final IDatabaseAdapter adapter, final String classifierWorkingName) throws SQLException {
		Map<String, Object> errorMap = new HashMap<>();
		errorMap.put("openmlid", openmlid);
		errorMap.put("datapoints", datapoints);
		if (!rowInBoundTable.isPresent()) {
			adapter.insert("executionbounds_" + classifierWorkingName, errorMap);
		} else if (datapoints < rowInBoundTable.get().getAsInt("datapoints")) {
			Map<String, Object> condMap = new HashMap<>();
			condMap.put("openmlid", openmlid);
			adapter.update("executionbounds_" + classifierWorkingName, errorMap, condMap);
		}
	}

}
