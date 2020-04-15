package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.io.IOException;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.commons.lang3.RandomStringUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.control.ILoggingCustomizable;
import org.api4.java.datastructure.kvstore.IKVStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;

import ai.libs.jaicore.basic.MathExt;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.timing.TimedComputation;
import tpami.basealgorithmlearning.datagathering.PeakMemoryObserver;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.SingleClassifierEnhancer;

public class DefaultMetaLearnerExperimentSetEvaluator implements IExperimentSetEvaluator, ILoggingCustomizable {

	private Logger logger = LoggerFactory.getLogger("DefaultMetaLearnerExperimentSetEvaluator");

	private Map<LeakingBaselearnerWrapper, LeakingBaselearnerEventStatistics> baselearnerToEventStatisticsMapTraining;
	private Map<LeakingBaselearnerWrapper, LeakingBaselearnerEventStatistics> baselearnerToEventStatisticsMapPrediction;

	private DefaultMetaLearnerConfigContainer container;
	private Class<?> metalearnerClass;
	private String metalearnerName;
	private Timeout timeout;
	private EventBus eventBus;
	private Map<Integer, Collection<ExperimentDBEntry>> knownFailedExperimentsOfDatasets = new HashMap<>();
	private Map<Integer, Long> timestampOfLastErrorQueriesPerDataset = new HashMap<>();
	private final String executorDetails;

	public DefaultMetaLearnerExperimentSetEvaluator(final DefaultMetaLearnerConfigContainer container, final Class<?> metalearnerClass, final Timeout timeout, final String executorDetails) {
		this.container = container;
		this.metalearnerClass = metalearnerClass;
		this.metalearnerName = metalearnerClass.getSimpleName().toLowerCase();
		this.timeout = timeout;
		this.eventBus = new EventBus("metalearners");
		this.eventBus.register(this);
		this.executorDetails = executorDetails;
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException {
		long starttime = System.currentTimeMillis();
		this.initializeNewExperiment();

		final PeakMemoryObserver mobs = new PeakMemoryObserver();
		mobs.start();

		this.logger.info("Reading in experiment with id {}.", experimentEntry.getId());
		Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
		int seed = Integer.parseInt(keys.get("seed"));
		int openmlid = Integer.parseInt(keys.get("openmlid"));
		int datapoints = Integer.parseInt(keys.get("datapoints"));
		String baselearner = keys.get("baselearner");

		this.logger.info("Running experiment {}, which is {} on dataset {} with seed {} and {} data points. Executor details: {}", experimentEntry.getId(), this.metalearnerClass.getName(), openmlid, seed, datapoints, this.executorDetails);
		Map<String, Object> map = new HashMap<>();
		map.put("executordetails", this.executorDetails);
		processor.processResults(map);
		map.clear();

		/* load dataset */
		List<ILabeledDataset<?>> splitTmp = null;
		IWekaClassifier metalearner = null;
		try {
			Dataset ds = (Dataset) OpenMLDatasetReader.deserializeDataset(openmlid);
			if (ds.getLabelAttribute() instanceof INumericAttribute) {
				this.logger.info("Converting numeric dataset to classification dataset!");
				ds = (Dataset) DatasetUtil.convertToClassificationDataset(ds);
			}

			/* check whether the dataset is reproducible */
			if (ds.getConstructionPlan().getInstructions().isEmpty()) {
				mobs.cancel();
				throw new IllegalStateException("Construction plan for dataset is empty!");
			}

			/* check whether we have reports that even smaller sizes do not work */
			Map<String, Object> comparisonExperiments = new HashMap<>();
			comparisonExperiments.put("openmlid", openmlid);
			this.checkFail(this.knownFailedExperimentsOfDatasets.get(openmlid), datapoints, baselearner);
			this.logger.info("Did not find any reason to believe that this experiment will fail based on earlier insights.");
			int MIN = 15;
			if (System.currentTimeMillis() - this.timestampOfLastErrorQueriesPerDataset.computeIfAbsent(openmlid, id -> (long)0) > 1000 * 60 * MIN) {
				this.logger.info("Last check was at least {} minutes ago, checking again.", MIN);
				Collection<ExperimentDBEntry> failedExperimentsOnThisDataset = this.container.getDatabaseHandle().getFailedExperiments(comparisonExperiments);
				this.knownFailedExperimentsOfDatasets.put(openmlid, failedExperimentsOnThisDataset);
				this.timestampOfLastErrorQueriesPerDataset.put(openmlid, System.currentTimeMillis());
				this.checkFail(failedExperimentsOnThisDataset, datapoints, baselearner);
				this.logger.info("Did not find any reason to believe that this experiment will fail. Running {} with options {} on dataset {} with seed {} and {} data points.", baselearner, keys.get("algorithmoptions"), openmlid, seed, datapoints);
			}
			else {
				this.logger.info("Last check was within last {} minutes ago, not checking but conducting (blindly trusting that this experiment is not dominated by some other finished meanwhile).", MIN);
			}

			this.logger.info("Label: {} ... {}", ds.getLabelAttribute().getClass().getName(), ds.getLabelAttribute().getStringDescriptionOfDomain());
			if (datapoints > ds.size()) {
				throw new IllegalStateException("Ddataset has not sufficient datapoints.");
			}
			double portion = datapoints * 1.0 / ds.size();
			splitTmp = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, portion);

			/* get json describing training and test data */
			//			JsonNode trainNode = om.valueToTree(((IReconstructible) splitTmp.get(0)).getConstructionPlan());
			//			JsonNode testNode = om.valueToTree(((IReconstructible) splitTmp.get(1)).getConstructionPlan());
			//			ArrayNode trainInstructions = (ArrayNode) trainNode.get("instructions");
			//			ArrayNode testInstructions = (ArrayNode) testNode.get("instructions");
			//			ArrayNode commonPrefix = om.createArrayNode();
			//			int numInstructions = trainInstructions.size();
			//			this.logger.info("Train instructions consist of {} instructions.", numInstructions);
			//			for (int i = 0; i < numInstructions - 1; i++) {
			//				if (!trainInstructions.get(i).equals(testInstructions.get(i))) {
			//					throw new IllegalStateException("Train and test data do not have common prefix!\nTrain: " + trainInstructions.get(i) + "\nTest: " + testInstructions.get(i));
			//				}
			//				commonPrefix.add(trainInstructions.get(i));
			//			}
			//			if (commonPrefix.size() == 0) {
			//				throw new IllegalStateException("The common prefix of train and test data must not be empty.\nTrain instructions:\n" + trainInstructions + "\nTest instructions:\n" + testInstructions);
			//			}

			/* create classifier */
			// ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(config, databaseHandle);
			// preparer.setLoggerName("example");
			// preparer.synchronizeExperiments();
			// Syst
			metalearner = new WekaClassifier(AbstractClassifier.forName(this.metalearnerClass.getName(), null));

			Classifier baseLearnerToUse = AbstractClassifier.forName(baselearner, null);
			this.logger.info("Using base learner {}", WekaUtil.getClassifierDescriptor(baseLearnerToUse));
			if (metalearner.getClassifier() instanceof SingleClassifierEnhancer) {
				((SingleClassifierEnhancer) metalearner.getClassifier()).setClassifier(new LeakingBaselearnerWrapper(this.eventBus, baseLearnerToUse, RandomStringUtils.random(80)));
			} else if (metalearner.getClassifier() instanceof MultipleClassifiersCombiner) {
				((MultipleClassifiersCombiner) metalearner.getClassifier()).setClassifiers(new AbstractClassifier[] { new LeakingBaselearnerWrapper(this.eventBus, baseLearnerToUse, RandomStringUtils.random(80)) });
			} else {
				mobs.cancel();
				throw new RuntimeException("Unknown super class of meta learner: " + metalearner.getClassifier().getClass());
			}

			/* now write core data about this experiment */
			//			map.put("evaluationinputdata", commonPrefix.toString());
			//			if (map.get("evaluationinputdata").equals("[]")) {
			//				throw new IllegalStateException();
			//			}
			//			map.put("traindata", trainInstructions.get(numInstructions - 1).toString());
			//			map.put("testdata", testInstructions.get(numInstructions - 1).toString());
			//			map.put("pipeline", om.writeValueAsString(((IReconstructible) metalearner).getConstructionPlan()));

			/* verify that the compose data recover to the same */
			//			ObjectNode composed = om.createObjectNode();
			//			composed.set("instructions", om.readTree(map.get("evaluationinputdata").toString()));
			//			ArrayNode an = (ArrayNode) composed.get("instructions");
			//			JsonNode trainDI = om.readTree(map.get("traindata").toString());
			//			JsonNode testDI = om.readTree(map.get("testdata").toString());
			//			an.add(trainDI);
			//			if (!om.readValue(composed.toString(), ReconstructionPlan.class).reconstructObject().equals(splitTmp.get(0))) {
			//				throw new IllegalStateException("Reconstruction of train data failed!");
			//			}
			//			an.remove(an.size() - 1);
			//			an.add(testDI);
			//			if (!om.readValue(composed.toString(), ReconstructionPlan.class).reconstructObject().equals(splitTmp.get(1))) {
			//				throw new IllegalStateException("Reconstruction of test data failed!");
			//			}
			//
			//			this.logger.info("{}", map);
			//			processor.processResults(map);
			//			map.clear();
		} catch (Throwable e) {
			mobs.cancel();
			throw new ExperimentEvaluationFailedException(e);
		}
		final List<ILabeledDataset<?>> split = splitTmp;
		final IWekaClassifier c = metalearner;

		/* now train classifier */
		long timePriorToTrainCommand = System.currentTimeMillis();
		this.logger.info("Experiment preparaion (including splits) finished after {}s", MathExt.round((timePriorToTrainCommand - starttime) / 1000.0, 2));
		SimpleDateFormat format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
		map.put("train_start", format.format(new Date(System.currentTimeMillis())));
		long deadlinetimestamp = timePriorToTrainCommand + this.timeout.milliseconds();
		try {
			mobs.reset();
			TimedComputation.compute(() -> {
				c.fit(split.get(0));
				return null;
			}, this.timeout.milliseconds(), "Experiment timeout exceeded.");
			Thread.sleep(1000);
		} catch (Throwable e) {
			map.put("train_end", format.format(new Date(System.currentTimeMillis())));
			processor.processResults(map);
			mobs.cancel();
			throw new ExperimentEvaluationFailedException(e);
		}
		map.put("train_end", format.format(new Date(System.currentTimeMillis())));
		map.put("memory_peak", mobs.getMaxMemoryConsumptionObserved());
		(((LeakingBaselearnerWrapper)((SingleClassifierEnhancer) metalearner.getClassifier()).getClassifier())).informThatMetaLearnerHasCompletedTraining();
		this.logger.info("Finished training, now testing on {} data points. Memory peak was {}", split.get(1).size(), map.get("memory_peak"));
		map.put("test_start", format.format(new Date(System.currentTimeMillis())));
		List<Integer> gt = new ArrayList<>();
		List<Integer> pr = new ArrayList<>();
		DescriptiveStatistics testRuntimeStats = new DescriptiveStatistics();
		try {
			long lastTimeoutCheck = 0;
			int n = split.get(1).size();
			for (ILabeledInstance i : split.get(1)) {
				if (System.currentTimeMillis() - lastTimeoutCheck > 10000) {
					lastTimeoutCheck = System.currentTimeMillis();
					long remainingTime = deadlinetimestamp - lastTimeoutCheck;
					this.logger.debug("Remaining time for this classifier: {}", remainingTime);
					if (remainingTime <= 0) {
						this.logger.info("Triggering timeout ...");
						throw new AlgorithmTimeoutedException(0);
					}
				}
				gt.add((int) i.getLabel());
				long predictionStart = System.currentTimeMillis();
				pr.add((int) c.predict(i).getPrediction());
				testRuntimeStats.addValue(System.currentTimeMillis() - predictionStart);
				if (gt.size() % 10000 == 0) {
					this.logger.info("{}/{} ({}%)", gt.size(), n, gt.size() * 100.0 / n);
				}
				else if (this.logger.isDebugEnabled()) {
					this.logger.debug("{}/{} ({}%)", gt.size(), n, gt.size() * 100.0 / n);
				}
			}
		} catch (Throwable e) {
			map.put("test_end", format.format(new Date(System.currentTimeMillis())));
			processor.processResults(map);
			throw new ExperimentEvaluationFailedException(e);
		}
		map.put("test_end", format.format(new Date(System.currentTimeMillis())));
		map.put("gt", gt);
		map.put("pr", pr);
		this.logger.info("Finished experiment #{}. Updating table. Here is the stats of the test runtimes:\n{}", experimentEntry.getId(), testRuntimeStats);
		processor.processResults(map);
		this.publishBaselearnerToEventStatisticsMapToDatabase(experimentEntry);
		mobs.cancel();
		this.logger.info("Finished Experiment {}. Results: {}. Additional information: {}", experimentEntry.getExperiment().getValuesOfKeyFields(), map, this.baselearnerToEventStatisticsMapTraining);
	}

	private void createAdditionalInformationTableIfNotExist() {
		List<IKVStore> resultSet = null;
		try {
			resultSet = this.container.getAdapter().query("SHOW TABLES LIKE 'additional_information_" + this.metalearnerName + "'");

			if (resultSet != null && resultSet.isEmpty()) {
				Map<String, String> fieldToTypeMap = new HashMap<>();
				fieldToTypeMap.put("info_id", "INT");
				fieldToTypeMap.put("experiment_id", "INT");
				fieldToTypeMap.put("openmlid", "VARCHAR(255)");
				fieldToTypeMap.put("datapoints", "INT");
				fieldToTypeMap.put("seed", "INT");
				fieldToTypeMap.put("baselearner", "VARCHAR(255)");
				fieldToTypeMap.put("hashCodeOfBaselearner", "LONGTEXT");
				for (String suffix : new String[] {"training", "prediction"}) {
					fieldToTypeMap.put("numberOfDistributionCalls_" + suffix, "INT");
					fieldToTypeMap.put("numberOfDistributionSCalls_" + suffix, "INT");
					fieldToTypeMap.put("numberOfClassifyInstanceCalls_" + suffix, "INT");
					fieldToTypeMap.put("numberOfBuildClassifierCalls_" + suffix, "INT");
					fieldToTypeMap.put("numberOfMetafeatureComputationCalls_" + suffix, "INT");
					fieldToTypeMap.put("firstDistributionTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("lastDistributionTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("firstDistributionSTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("lastDistributionSTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("firstClassifyInstanceTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("lastClassifyInstanceTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("firstBuildClassifierTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("lastBuildClassifierTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("firstMetafeatureTimestamp_" + suffix, "BIGINT(12)");
					fieldToTypeMap.put("lastMetafeatureTimestamp_" + suffix, "BIGINT(12)");
				}
				fieldToTypeMap.put("datasetMetafeatures", "LONGTEXT");

				this.container.getAdapter().createTable("additional_information_" + this.metalearnerName, "info_id",
						Arrays.asList("experiment_id", "openmlid", "datapoints", "seed", "baselearner", "hashCodeOfBaselearner", "numberOfDistributionCalls_training", "numberOfDistributionSCalls_training", "numberOfClassifyInstanceCalls_training",
								"numberOfBuildClassifierCalls_training", "numberOfMetafeatureComputationCalls_training", "firstDistributionTimestamp_training", "lastDistributionTimestamp_training", "firstDistributionSTimestamp_training", "lastDistributionSTimestamp_training",
								"firstClassifyInstanceTimestamp_training", "lastClassifyInstanceTimestamp_training", "firstBuildClassifierTimestamp_training", "lastBuildClassifierTimestamp_training", "firstMetafeatureTimestamp_training", "lastMetafeatureTimestamp_training", "numberOfDistributionCalls_prediction", "numberOfDistributionSCalls_prediction", "numberOfClassifyInstanceCalls_prediction",
								"numberOfBuildClassifierCalls_prediction", "numberOfMetafeatureComputationCalls_prediction", "firstDistributionTimestamp_prediction", "lastDistributionTimestamp_prediction", "firstDistributionSTimestamp_prediction", "lastDistributionSTimestamp_prediction",
								"firstClassifyInstanceTimestamp_prediction", "lastClassifyInstanceTimestamp_prediction", "firstBuildClassifierTimestamp_prediction", "lastBuildClassifierTimestamp_prediction", "firstMetafeatureTimestamp_prediction", "lastMetafeatureTimestamp_prediction",
								"datasetMetafeatures"),
						fieldToTypeMap, Arrays.asList("info_id", "experiment_id"));
			}
		} catch (SQLException | IOException e) {
			this.logger.error("Could not create table for additional information.", e);
		}
	}

	public void publishBaselearnerToEventStatisticsMapToDatabase(final ExperimentDBEntry experimentEntry) {
		this.createAdditionalInformationTableIfNotExist();
		this.logger.info("Publishing {} experiment results for experiment #{} to additional information table.", this.baselearnerToEventStatisticsMapTraining.size(), experimentEntry.getId());
		for (Entry<LeakingBaselearnerWrapper, LeakingBaselearnerEventStatistics> entry : this.baselearnerToEventStatisticsMapTraining.entrySet()) {
			Map<String, Object> insertableMap = entry.getValue().getAsInsertableMap("training");
			insertableMap.putAll(this.baselearnerToEventStatisticsMapPrediction.get(entry.getKey()).getAsInsertableMap("prediction"));

			insertableMap.put("hashCodeOfBaselearner", entry.getValue().getHashCodeOfBaselearner());
			insertableMap.put("datasetMetafeatures", entry.getValue().getDatasetMetafeatures().toString());

			int seed = Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("seed"));
			int openmlid = Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("openmlid"));
			int datapoints = Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("datapoints"));

			insertableMap.put("baselearner", experimentEntry.getExperiment().getValuesOfKeyFields().get("baselearner"));
			insertableMap.put("experiment_id", experimentEntry.getId());
			insertableMap.put("openmlid", openmlid);
			insertableMap.put("datapoints", datapoints);
			insertableMap.put("seed", seed);
			try {
				this.container.getAdapter().insert("additional_information_" + this.metalearnerName, insertableMap);
			} catch (SQLException e) {
				this.logger.error("Could not write additional information to database.", e);
			}
		}
		this.logger.info("Publication of additional information for experiment #{} finished.", experimentEntry.getId());
	}

	@Subscribe
	public void parseLeakingBaselearnerEvent(final LeakingBaselearnerEvent event) {
		this.logger.debug("Received event: {} - {}", event.getTimestamp(), event.getEventType());
		this.forwardEventToRelevantStatisticsObject(event);
	}

	private void forwardEventToRelevantStatisticsObject(final LeakingBaselearnerEvent event) {
		LeakingBaselearnerWrapper wrapper = event.getLeakingBaselearnerWrapper();
		Map<LeakingBaselearnerWrapper, LeakingBaselearnerEventStatistics> relevantMap = event.isMetaLearnerTrained() ? this.baselearnerToEventStatisticsMapPrediction : this.baselearnerToEventStatisticsMapTraining;
		relevantMap.computeIfAbsent(wrapper, w -> new LeakingBaselearnerEventStatistics(w)).parseEvent(event);
	}

	private void initializeNewExperiment() {
		this.baselearnerToEventStatisticsMapTraining = new HashMap<>();
		this.baselearnerToEventStatisticsMapPrediction = new HashMap<>();
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}

	/* We can omit this execution in any of the following cases:
	 * - an earlier execution with less datapoints on the same dataset was canceled because there are not enough datapoints
	 * - an earlier execution with less datapoints on the same dataset AND the same algorithm options has timed out
	 **/
	public void checkFail(final Collection<ExperimentDBEntry> failedExperimentsOnThisDataset, final int datapoints, final String baselearner) throws ExperimentFailurePredictionException {
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
			boolean otherHasSameBaseLearner = e.getExperiment().getValuesOfKeyFields().get("baselearner").equals(baselearner);
			if (otherHasSameBaseLearner && requiresAtLeastAsManyPointsThanFailed && otherTimedOut) {
				reasonString.set("This has at least as much instances as " + idOfOther + ", which failed due to a timeout.");
				return true;
			}
			return false;
		});
		if (willFail) {
			this.logger.warn("Announcing fail with reason: {}", reasonString.get());
			throw new ExperimentFailurePredictionException("Experiment will fail for the following reason: " + reasonString.get());
		}
	}
}
