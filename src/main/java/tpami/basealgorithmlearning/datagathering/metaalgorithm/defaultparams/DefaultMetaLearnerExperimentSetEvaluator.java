package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.io.IOException;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
import ai.libs.jaicore.experiments.ExperimentDBEntry;
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
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.SingleClassifierEnhancer;

public class DefaultMetaLearnerExperimentSetEvaluator implements IExperimentSetEvaluator {

	private static final Logger LOGGER = LoggerFactory.getLogger("DefaultMetaLearnerExperimentSetEvaluator");

	private Map<LeakingBaselearnerWrapper, LeakingBaselearnerEventStatistics> baselearnerToEventStatisticsMap;

	private DefaultMetaLearnerConfigContainer container;
	private Class<?> metalearnerClass;
	private String metalearnerName;
	private Class<?> baselearnerClass;
	private String baselearnerName;
	private Timeout timeout;
	private EventBus eventBus;

	public DefaultMetaLearnerExperimentSetEvaluator(DefaultMetaLearnerConfigContainer container, Class<?> metalearnerClass, Class<?> baselearnerClass, Timeout timeout) {
		this.container = container;
		this.metalearnerClass = metalearnerClass;
		this.metalearnerName = metalearnerClass.getSimpleName().toLowerCase();
		this.baselearnerClass = baselearnerClass;
		this.baselearnerName = baselearnerClass.getSimpleName().toLowerCase();
		this.timeout = timeout;
		this.eventBus = new EventBus("metalearners");
		eventBus.register(this);
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException {
		initializeNewExperiment();

		final ObjectMapper om = new ObjectMapper();

		final PeakMemoryObserver mobs = new PeakMemoryObserver();
		mobs.start();

		LOGGER.info("Reading in experiment.");
		Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
		int seed = Integer.parseInt(keys.get("seed"));
		int openmlid = Integer.parseInt(keys.get("openmlid"));
		int datapoints = Integer.parseInt(keys.get("datapoints"));

		LOGGER.info("Running {} on dataset {} with seed {} and {} data points.", metalearnerClass.getName(), openmlid, seed, datapoints);
		Map<String, Object> map = new HashMap<>();

		/* load dataset */
		List<ILabeledDataset<?>> splitTmp = null;
		IWekaClassifier metalearner = null;
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
			// rowInBoundTable = container.getAdapter().getRowsOfTable("executionbounds_" + metalearnerName).stream().filter(r -> r.getAsInt("openmlid") == openmlid).findAny();
			// LOGGER.info("Bound available: {}", (rowInBoundTable.isPresent() ? "yes (" + rowInBoundTable.get().getAsInt("datapoints") + ")" : "no"));
			// if (rowInBoundTable.isPresent() && rowInBoundTable.get().getAsInt("datapoints") < datapoints) {
			// throw new IllegalStateException("Experiment canceled due to lower bound on other experiments on this dataset.");
			// }

			LOGGER.info("Label: {} ... {}", ds.getLabelAttribute().getClass().getName(), ds.getLabelAttribute().getStringDescriptionOfDomain());
			if (datapoints > ds.size()) {
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
			metalearner = new WekaClassifier(AbstractClassifier.forName(metalearnerClass.getName(), null));

			Classifier baseLearnerToUse = AbstractClassifier.forName(baselearnerClass.getName(), null);
			if (metalearner.getClassifier() instanceof SingleClassifierEnhancer) {
				((SingleClassifierEnhancer) metalearner.getClassifier()).setClassifier(new LeakingBaselearnerWrapper(eventBus, baseLearnerToUse));
			} else if (metalearner.getClassifier() instanceof MultipleClassifiersCombiner) {
				((MultipleClassifiersCombiner) metalearner.getClassifier()).setClassifiers(new AbstractClassifier[] { new LeakingBaselearnerWrapper(eventBus, baseLearnerToUse) });
			} else {
				throw new RuntimeException("Unknown super class of meta learner: " + metalearner.getClassifier().getClass());
			}

			/* now write core data about this experiment */
			map.put("evaluationinputdata", commonPrefix.toString());
			if (map.get("evaluationinputdata").equals("[]")) {
				throw new IllegalStateException();
			}
			map.put("traindata", trainInstructions.get(numInstructions - 1).toString());
			map.put("testdata", testInstructions.get(numInstructions - 1).toString());
			map.put("pipeline", om.writeValueAsString(((IReconstructible) metalearner).getConstructionPlan()));

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
		final IWekaClassifier c = metalearner;

		/* now train classifier */
		SimpleDateFormat format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
		map.put("train_start", format.format(new Date(System.currentTimeMillis())));
		long deadlinetimestamp = System.currentTimeMillis() + timeout.milliseconds();
		try {
			mobs.reset();
			TimedComputation.compute(() -> {
				c.fit(split.get(0));
				return null;
			}, timeout.milliseconds(), "Experiment timeout exceeded.");
			Thread.sleep(1000);
		} catch (Throwable e) {
			map.put("train_end", format.format(new Date(System.currentTimeMillis())));
			processor.processResults(map);
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
		publishBaselearnerToEventStatisticsMapToDatabase(experimentEntry);
		LOGGER.info("Finished Experiment {}. Results: {}. Additional information: {}", experimentEntry.getExperiment().getValuesOfKeyFields(), map, baselearnerToEventStatisticsMap);

	}

	private void createAdditionalInformationTableIfNotExist() {
		List<IKVStore> resultSet = null;
		try {
			resultSet = container.getAdapter().query("SHOW TABLES LIKE 'additional_information_" + metalearnerName + "'");

			if (resultSet != null && resultSet.isEmpty()) {
				Map<String, String> fieldToTypeMap = new HashMap<>();
				fieldToTypeMap.put("info_id", "INT");
				fieldToTypeMap.put("experiment_id", "INT");
				fieldToTypeMap.put("openmlid", "VARCHAR(255)");
				fieldToTypeMap.put("datapoints", "INT");
				fieldToTypeMap.put("seed", "INT");
				fieldToTypeMap.put("baselearner", "VARCHAR(255)");
				fieldToTypeMap.put("hashCodeOfBaselearner", "LONGTEXT");
				fieldToTypeMap.put("numberOfDistributionCalls", "INT");
				fieldToTypeMap.put("numberOfDistributionSCalls", "INT");
				fieldToTypeMap.put("numberOfClassifyInstanceCalls", "INT");
				fieldToTypeMap.put("numberOfBuildClassifierCalls", "INT");
				fieldToTypeMap.put("numberOfMetafeatureComputationCalls", "INT");
				fieldToTypeMap.put("firstDistributionTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("lastDistributionTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("firstDistributionSTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("lastDistributionSTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("firstClassifyInstanceTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("lastClassifyInstanceTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("firstBuildClassifierTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("lastBuildClassifierTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("firstMetafeatureTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("lastMetafeatureTimestamp", "BIGINT(12)");
				fieldToTypeMap.put("datasetMetafeatures", "LONGTEXT");

				container.getAdapter().createTable("additional_information_" + metalearnerName, "info_id",
						Arrays.asList("experiment_id", "openmlid", "datapoints", "seed", "baselearner", "hashCodeOfBaselearner", "numberOfDistributionCalls", "numberOfDistributionSCalls", "numberOfClassifyInstanceCalls",
								"numberOfBuildClassifierCalls", "numberOfMetafeatureComputationCalls", "firstDistributionTimestamp", "lastDistributionTimestamp", "firstDistributionSTimestamp", "lastDistributionSTimestamp",
								"firstClassifyInstanceTimestamp", "lastClassifyInstanceTimestamp", "firstBuildClassifierTimestamp", "lastBuildClassifierTimestamp", "firstMetafeatureTimestamp", "lastMetafeatureTimestamp",
								"datasetMetafeatures"),
						fieldToTypeMap, Arrays.asList("info_id", "experiment_id"));
			}
		} catch (SQLException | IOException e) {
			LOGGER.error("Could not create table for additional information.", e);
		}
	}

	public void publishBaselearnerToEventStatisticsMapToDatabase(ExperimentDBEntry experimentEntry) {
		createAdditionalInformationTableIfNotExist();
		for (Entry<LeakingBaselearnerWrapper, LeakingBaselearnerEventStatistics> entry : baselearnerToEventStatisticsMap.entrySet()) {
			Map<String, Object> insertableMap = entry.getValue().getAsInsertableMap();

			int seed = Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("seed"));
			int openmlid = Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("openmlid"));
			int datapoints = Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("datapoints"));

			insertableMap.put("baselearner", baselearnerName);
			insertableMap.put("experiment_id", experimentEntry.getId());
			insertableMap.put("openmlid", openmlid);
			insertableMap.put("datapoints", datapoints);
			insertableMap.put("seed", seed);
			try {
				container.getAdapter().insert("additional_information_" + metalearnerName, insertableMap);
			} catch (SQLException e) {
				LOGGER.error("Could not write additional information to database.", e);
			}
		}

	}

	@Subscribe
	public void parseLeakingBaselearnerEvent(LeakingBaselearnerEvent event) {
		LOGGER.debug("Received event: " + event.getTimestamp() + " - " + event.getEventType());
		forwardEventToRelevantStatisticsObject(event);
	}

	private void forwardEventToRelevantStatisticsObject(LeakingBaselearnerEvent event) {
		LeakingBaselearnerWrapper wrapper = event.getLeakingBaselearnerWrapper();
		if (!baselearnerToEventStatisticsMap.containsKey(wrapper)) {
			baselearnerToEventStatisticsMap.put(wrapper, new LeakingBaselearnerEventStatistics(wrapper));
		}
		baselearnerToEventStatisticsMap.get(wrapper).parseEvent(event);
	}

	private void initializeNewExperiment() {
		baselearnerToEventStatisticsMap = new HashMap<>();
	}

}
