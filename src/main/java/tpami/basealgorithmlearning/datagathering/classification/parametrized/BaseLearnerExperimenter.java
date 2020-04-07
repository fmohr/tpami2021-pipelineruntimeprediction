package tpami.basealgorithmlearning.datagathering.classification.parametrized;
import java.sql.Date;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.timing.TimedComputation;
import tpami.basealgorithmlearning.datagathering.PeakMemoryObserver;
import weka.classifiers.AbstractClassifier;
import weka.core.OptionHandler;

public class BaseLearnerExperimenter {

	private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");

	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout to = new Timeout(1, TimeUnit.HOURS);
	private static BaseLearnerConfigContainer container;
	private static IExperimentDatabaseHandle databaseHandle;

	private static Map<Integer, Collection<ExperimentDBEntry>> knownFailedExperimentsOfDatasets = new HashMap<>();
	private static Map<Integer, Long> timestampOfLastErrorQueriesPerDataset = new HashMap<>();

	public static void main(final String[] args) throws Exception {

		if (args.length != 2) {
			LOGGER.error("Provide exactly two arguments to experiment runner:\n\t#1: database config file\n\t#2: class name of WEKA classifier");
			return;
		}

		container = new BaseLearnerConfigContainer(args[0], args[1]);
		Class<?> classifierClass = Class.forName(args[1]);
		databaseHandle = container.getDatabaseHandle();

		/* run an experiment */
		final ObjectMapper om = new ObjectMapper();

		final PeakMemoryObserver mobs = new PeakMemoryObserver();
		mobs.start();

		LOGGER.info("Creating the runner.");
		ExperimentRunner runner = new ExperimentRunner(container.getConfig(), new IExperimentSetEvaluator() {

			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException, ExperimentFailurePredictionException {
				try {

					LOGGER.info("Reading in experiment with id {}.", experimentEntry.getId());
					Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
					int seed = Integer.parseInt(keys.get("seed"));
					int openmlid = Integer.parseInt(keys.get("openmlid"));
					int datapoints = Integer.parseInt(keys.get("datapoints"));
					String[] options = keys.get("algorithmoptions").split(" ");

					/* first of all, check whether we can omit execution due to fails of similar earlier runs. */
					Map<String, Object> comparisonExperiments = new HashMap<>();
					comparisonExperiments.put("openmlid", openmlid);
					checkFail(knownFailedExperimentsOfDatasets.get(openmlid), datapoints, keys.get("algorithmoptions"));
					LOGGER.info("Did not find any reason to believe that this experiment will fail based on earlier insights.");
					int MIN = 15;
					if (System.currentTimeMillis() - timestampOfLastErrorQueriesPerDataset.computeIfAbsent(openmlid, id -> (long)0) > 1000 * 60 * MIN) {
						LOGGER.info("Last check was at least {} minutes ago, checking again.", MIN);
						Collection<ExperimentDBEntry> failedExperimentsOnThisDataset = databaseHandle.getFailedExperiments(comparisonExperiments);
						knownFailedExperimentsOfDatasets.put(openmlid, failedExperimentsOnThisDataset);
						timestampOfLastErrorQueriesPerDataset.put(openmlid, System.currentTimeMillis());
						checkFail(failedExperimentsOnThisDataset, datapoints, keys.get("algorithmoptions"));
						LOGGER.info("Did not find any reason to believe that this experiment will fail. Running {} with options {} on dataset {} with seed {} and {} data points.", classifierClass.getName(), keys.get("algorithmoptions"), openmlid, seed, datapoints);
					}
					else {
						LOGGER.info("Last check was within last {} minutes ago, not checking but conducting (blindly trusting that this experiment is not dominated by some other finished meanwhile).", MIN);
					}

					Map<String, Object> map = new HashMap<>();

					/* load dataset */
					List<ILabeledDataset<?>> splitTmp = null;
					IWekaClassifier cTmp = null;
					Dataset ds = (Dataset)OpenMLDatasetReader.deserializeDataset(openmlid);
					if (ds.getLabelAttribute() instanceof INumericAttribute) {
						LOGGER.info("Converting numeric dataset to classification dataset!");
						ds = (Dataset)DatasetUtil.convertToClassificationDataset(ds);
					}

					/* check whether the dataset is reproducible */
					if (ds.getConstructionPlan().getInstructions().isEmpty()) {
						throw new IllegalStateException("Construction plan for dataset is empty!");
					}

					LOGGER.info("Label: {} ... {}", ds.getLabelAttribute().getClass().getName(), ds.getLabelAttribute().getStringDescriptionOfDomain());
					if (datapoints > ds.size()) {
						throw new IllegalStateException("Invalid experiment. The dataset has not sufficient datapoints.");
					}
					double portion = datapoints * 1.0 / ds.size();
					splitTmp = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, portion);

					/* get json describing training and test data */
					//					JsonNode trainNode = om.valueToTree(((IReconstructible)splitTmp.get(0)).getConstructionPlan());
					//					JsonNode testNode = om.valueToTree(((IReconstructible)splitTmp.get(1)).getConstructionPlan());
					//					ArrayNode trainInstructions = (ArrayNode)trainNode.get("instructions");
					//					ArrayNode testInstructions = (ArrayNode)testNode.get("instructions");
					//					ArrayNode commonPrefix = om.createArrayNode();
					//					int numInstructions = trainInstructions.size();
					//					LOGGER.info("Train instructions consist of {} instructions.", numInstructions);
					//					for (int i = 0; i < numInstructions - 1; i++) {
					//						if (!trainInstructions.get(i).equals(testInstructions.get(i))) {
					//							throw new IllegalStateException("Train and test data do not have common prefix!\nTrain: " + trainInstructions.get(i) + "\nTest: " + testInstructions.get(i));
					//						}
					//						commonPrefix.add(trainInstructions.get(i));
					//					}
					//					if (commonPrefix.size() == 0) {
					//						throw new IllegalStateException("The common prefix of train and test data must not be empty.\nTrain instructions:\n" + trainInstructions + "\nTest instructions:\n" + testInstructions);
					//					}

					/* create classifier */
					cTmp = new WekaClassifier(AbstractClassifier.forName(classifierClass.getName(), options));
					LOGGER.info("Actual option description of the created classifier: {}", Arrays.toString(((OptionHandler)cTmp.getClassifier()).getOptions()));

					//					/* now write core data about this experiment */
					//					map.put("evaluationinputdata", commonPrefix.toString());
					//					if (map.get("evaluationinputdata").equals("[]")) {
					//						throw new IllegalStateException();
					//					}
					//					map.put("traindata", trainInstructions.get(numInstructions - 1).toString());
					//					map.put("testdata", testInstructions.get(numInstructions - 1).toString());
					//					map.put("pipeline", om.writeValueAsString(((IReconstructible)cTmp).getConstructionPlan()));
					//
					//					/* verify that the compose data recover to the same */
					//					ObjectNode composed = om.createObjectNode();
					//					composed.set("instructions", om.readTree(map.get("evaluationinputdata").toString()));
					//					ArrayNode an = (ArrayNode)composed.get("instructions");
					//					JsonNode trainDI = om.readTree(map.get("traindata").toString());
					//					JsonNode testDI = om.readTree(map.get("testdata").toString());
					//					an.add(trainDI);
					//					if (!om.readValue(composed.toString(), ReconstructionPlan.class).reconstructObject().equals(splitTmp.get(0))) {
					//						throw new IllegalStateException("Reconstruction of train data failed!");
					//					}
					//					an.remove(an.size() - 1);
					//					an.add(testDI);
					//					if (!om.readValue(composed.toString(), ReconstructionPlan.class).reconstructObject().equals(splitTmp.get(1))) {
					//						throw new IllegalStateException("Reconstruction of test data failed!");
					//					}
					//
					//					LOGGER.info("{}", map);
					//					processor.processResults(map);
					//					map.clear();

					final List<ILabeledDataset<?>> split = splitTmp;
					final IWekaClassifier c = cTmp;

					/* now train classifier */
					map.put("train_start",  DATE_FORMAT.format(new Date(System.currentTimeMillis())));
					long deadlinetimestamp = System.currentTimeMillis() + to.milliseconds();
					try {
						mobs.reset();
						TimedComputation.compute(() -> { c.fit(split.get(0)); return null;}, to.milliseconds(), "Experiment timeout exceeded.");
						Thread.sleep(1000);
					} catch (Throwable e) {
						map.put("train_end",  DATE_FORMAT.format(new Date(System.currentTimeMillis())));
						processor.processResults(map);
						throw e;
					}
					map.put("train_end",  DATE_FORMAT.format(new Date(System.currentTimeMillis())));
					map.put("memory_peak", mobs.getMaxMemoryConsumptionObserved());
					processor.processResults(map);
					map.clear();
					LOGGER.info("Finished training, now testing. Memory peak was {}", map.get("memory_peak"));
					map.put("test_start",  DATE_FORMAT.format(new Date(System.currentTimeMillis())));
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
								if (remainingTime <=  0) {
									LOGGER.info("Triggering timeout ...");
									throw new AlgorithmTimeoutedException(0);
								}
							}
							gt.add((int)i.getLabel());
							pr.add((int)c.predict(i).getPrediction());
							LOGGER.debug("{}/{} ({}%)", gt.size(), n, gt.size() * 100.0 / n);
						}
					}
					catch (Throwable e) {
						map.put("test_end",  DATE_FORMAT.format(new Date(System.currentTimeMillis())));
						processor.processResults(map);
						throw e;
					}
					map.put("test_end", DATE_FORMAT.format(new Date(System.currentTimeMillis())));
					map.put("gt", gt);
					map.put("pr", pr);
					processor.processResults(map);
					LOGGER.info("Finished Experiment {}. Results: {}", experimentEntry.getExperiment().getValuesOfKeyFields(),  map);

				}
				catch (ExperimentFailurePredictionException e) {
					throw e;
				}
				catch (Throwable e) {
					throw new ExperimentEvaluationFailedException(e);
				}
			}}, databaseHandle);

		LOGGER.info("Runner created.");

		LOGGER.info("Running random experiments with timeout " + TOTAL_RUNTIME_IN_SECONDS + "s.");
		runner.setLoggerName("example");
		long start = System.currentTimeMillis();
		while ((System.currentTimeMillis() - start) / 1000 <= TOTAL_RUNTIME_IN_SECONDS - to.seconds() * 1.1) {
			LOGGER.info("Conducting next experiment.");
			try {
				runner.randomlyConductExperiments(1);
			}
			catch (Throwable e) {
				e.printStackTrace();
			}
		}
		LOGGER.info("No more time left to conduct more experiments. Stopping.");
		System.exit(0);
	}

	/* We can omit this execution in any of the following cases:
	 * - an earlier execution with less datapoints on the same dataset was canceled because there are not enough datapoints
	 * - an earlier execution with less datapoints on the same dataset AND the same algorithm options has timed out
	 **/
	public static void checkFail(final Collection<ExperimentDBEntry> failedExperimentsOnThisDataset, final int datapoints, final String algorithmoptions) throws ExperimentFailurePredictionException {
		if (failedExperimentsOnThisDataset == null) {
			return;
		}
		AtomicReference<String> reasonString = new AtomicReference<>();
		boolean willFail = failedExperimentsOnThisDataset.stream().anyMatch(e -> {
			int idOfOther = e.getId();
			int numInstancesRequiredByOthers = Integer.parseInt(e.getExperiment().getValuesOfKeyFields().get("datapoints"));
			boolean requiresAtLeastAsManyPointsThanFailed =  numInstancesRequiredByOthers <= datapoints;
			String errorMsg = e.getExperiment().getError().toLowerCase();
			boolean otherFailedDueToTooFewInstances = errorMsg.contains("dataset has not sufficient datapoints");
			if (requiresAtLeastAsManyPointsThanFailed && otherFailedDueToTooFewInstances) {
				reasonString.set("This experiment requires at least as many instances as " + idOfOther + ", which failed because it demanded too many instances (" + numInstancesRequiredByOthers + ", and here we require " +  datapoints + ").");
				return true;
			}
			boolean otherTimedOut = errorMsg.contains("timeout");
			boolean otherHasSameOptions = e.getExperiment().getValuesOfKeyFields().get("algorithmoptions").equals(algorithmoptions);
			if (otherHasSameOptions && requiresAtLeastAsManyPointsThanFailed && otherTimedOut) {
				reasonString.set("This has at least as much instances as " + idOfOther + ", which failed due to a timeout.");
				return true;
			}
			return false;
		});
		if (willFail) {
			LOGGER.warn("Announcing fail with reason: {}", reasonString.get());
			throw new ExperimentFailurePredictionException("Experiment will fail for the following reason: " + reasonString.get());
		}
	}
}
