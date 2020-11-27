package tpami;

import java.io.File;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigCache;
import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.IPredictionAndGroundTruthTable;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.core.learner.ISupervisedLearner;
import org.api4.java.common.reconstruction.IReconstructible;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.google.common.eventbus.Subscribe;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.classification.loss.dataset.EAggregatedClassifierMetric;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.evaluation.evaluator.events.TrainTestSplitEvaluationCompletedEvent;
import ai.libs.jaicore.ml.core.evaluation.evaluator.events.TrainTestSplitEvaluationFailedEvent;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.classification.pipeline.MLPipeline;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;

public class Experimenter {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");

	public static void main(final String[] args) throws Exception {
		/* parse and interpret input */
		final int k = Integer.parseInt(args[0]);
		// for (int k = 0; k < 20; k++) {
		List<Integer> dsLines = FileUtil.readFileAsList(new File("datasets.conf")).stream().map(l -> Integer.parseInt(l.split("//")[0].trim())).collect(Collectors.toList());
		final Integer[] datasets = dsLines.toArray(new Integer[0]);
		final int seed = (int) Math.floor(k * 1.0 / datasets.length);
		final int dataset = k % datasets.length;
		final int openMLId = datasets[dataset];
		System.out.println("Running ML-Plan on dataset " + openMLId + " with seed " + seed);

		ILabeledDataset<?> ds = OpenMLDatasetReader.deserializeDataset(openMLId);
		if (ds.getLabelAttribute() instanceof INumericAttribute) {
			System.out.println("Converting numeric dataset to classification dataset!");
			ds = DatasetUtil.convertToClassificationDataset(ds);
		}

		System.out.println("Label: " + ds.getLabelAttribute().getClass().getName() + " ... " + ds.getLabelAttribute().getStringDescriptionOfDomain());

		/* initialize mlplan */
		MLPlanWekaBuilder builder = new MLPlanWekaBuilder().withAlgorithmConfigFile(new File("mlplan-weka-eval.properties")).withSeed(seed);

		List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, .7);

		MLPlan<IWekaClassifier> mlplan = builder.withDataset(split.get(0)).build();
		mlplan.setLoggerName("testedalgorithm");
		// if (true) {
		// continue;
		// }
		/* setup database connection */
		IDatabaseConfig dbConfig = (IDatabaseConfig) ConfigCache.getOrCreate(IDatabaseConfig.class).loadPropertiesFromFile(new File("dbcon.conf"));
		try (IDatabaseAdapter dbAdapter = DatabaseAdapterFactory.get(dbConfig)) {
			final String table = dbConfig.getDBTableName();
			final String resultTable = table + "_results";

			/* register a listener  */
			mlplan.registerListener(new Object() {

				public Map<String, Object> getReportMap(final ISupervisedLearner<?, ?> learner, final ILearnerRunReport r) {
					Map<String, Object> map = new HashMap<>();
					ObjectMapper om = new ObjectMapper();
					if (!(r.getTrainSet() instanceof IReconstructible)) {
						throw new IllegalArgumentException("Train data is not reconstructible.");
					}
					if (!(r.getTestSet() instanceof IReconstructible)) {
						throw new IllegalArgumentException("Test data is not reconstructible.");
					}
					if (!(learner instanceof IReconstructible)) {
						throw new IllegalArgumentException("Learner " + learner.getClass().getName() + " is not reconstructible.");
					}
					// map.put("testdata", r.getTestSet());
					try {

						/* get json describing training and test data */
						JsonNode trainNode = om.valueToTree(((IReconstructible) r.getTrainSet()).getConstructionPlan());
						JsonNode testNode = om.valueToTree(((IReconstructible) r.getTestSet()).getConstructionPlan());
						ArrayNode trainInstructions = (ArrayNode) trainNode.get("instructions");
						ArrayNode testInstructions = (ArrayNode) testNode.get("instructions");
						ArrayNode commonPrefix = om.createArrayNode();
						int numInstructions = trainInstructions.size();
						for (int i = 0; i < numInstructions - 1; i++) {
							if (!trainInstructions.get(i).equals(testInstructions.get(i))) {
								throw new IllegalStateException("Train and test data do not have common prefix!\nTrain: " + trainInstructions.get(i) + "\nTest: " + testInstructions.get(i));
							}
							commonPrefix.add(trainInstructions.get(i));
						}

						map.put("openmlid", openMLId);
						map.put("seed", seed);
						map.put("evaluationinputdata", commonPrefix.toString());
						map.put("traindata", trainInstructions.get(numInstructions - 1).toString());
						map.put("testdata", testInstructions.get(numInstructions - 1).toString());
						map.put("pipeline", om.writeValueAsString(((IReconstructible) learner).getConstructionPlan()));
					} catch (JsonProcessingException e) {
						e.printStackTrace();
					}
					SimpleDateFormat format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
					map.put("train_start", format.format(new Date(r.getTrainStartTime())));
					map.put("train_end", format.format(new Date(r.getTrainEndTime())));
					if (r.getTestStartTime() > 0) {
						map.put("test_start", format.format(new Date(r.getTestStartTime())));
						map.put("test_end", format.format(new Date(r.getTestEndTime())));
					}
					if (r.getException() != null) {
						map.put("exception", LoggerUtil.getExceptionInfo(r.getException()));
					}
					IPredictionAndGroundTruthTable<?, ?> diffList = r.getPredictionDiffList();
					if (diffList != null) {
						try {
							map.put("gt", om.writeValueAsString(diffList.getGroundTruthAsList()));
							map.put("pr", om.writeValueAsString(diffList.getPredictionsAsList()));
						} catch (JsonProcessingException e) {
							e.printStackTrace();
						}
					}
					return map;
				}

				@Subscribe
				public void receiveEvent(final TrainTestSplitEvaluationFailedEvent e) throws SQLException { // this event is fired whenever any pipeline is evaluated successfully
					WekaClassifier learner = (WekaClassifier) e.getLearner();
					if (learner.getClassifier() instanceof MLPipeline) {
						MLPipeline pipeline = ((MLPipeline) ((WekaClassifier) e.getLearner()).getClassifier());
						LOGGER.info("Received exception for learner {}: {}. Interrupt state is {}", pipeline, e.getReport().getException().getClass().getName(), Thread.currentThread().isInterrupted());
					} else {
						LOGGER.info("Received exception for learner {} after a training time of {}ms: {}. Interrupt state is {}", WekaUtil.getClassifierDescriptor(learner.getClassifier()),
								e.getReport().getTrainEndTime() - e.getReport().getTrainStartTime(), e.getReport().getException().getClass().getName(), Thread.currentThread().isInterrupted());
					}
					Map<String, Object> map = this.getReportMap(e.getLearner(), e.getReport());
					dbAdapter.insert(table, map);
				}

				@Subscribe
				public void receiveEvent(final TrainTestSplitEvaluationCompletedEvent e) throws SQLException { // this event is fired whenever any pipeline is evaluated successfully
					List<IPredictionAndGroundTruthTable<? extends Integer, ? extends ISingleLabelClassification>> l = Arrays.asList(e.getReport()).stream()
							.map(r -> (IPredictionAndGroundTruthTable<? extends Integer, ? extends ISingleLabelClassification>) r.getPredictionDiffList()).collect(Collectors.toList());
					double errorRate = EAggregatedClassifierMetric.MEAN_ERRORRATE.loss(l);
					WekaClassifier learner = (WekaClassifier) e.getLearner();
					if (learner.getClassifier() instanceof MLPipeline) {
						MLPipeline pipeline = ((MLPipeline) ((WekaClassifier) e.getLearner()).getClassifier());
						LOGGER.info("Received single evaluation error rate for pipeline {} is {}. Interrupt state is {}", pipeline, errorRate, Thread.currentThread().isInterrupted());
					} else {
						LOGGER.info("Received single evaluation error rate for learner {} is {}. Interrupt state is {}", WekaUtil.getDescriptor(learner.getClassifier()), errorRate, Thread.currentThread().isInterrupted());
					}
					Map<String, Object> map = this.getReportMap(e.getLearner(), e.getReport());
					dbAdapter.insert(table, map);
				}
			});

			// mlplan.setLoggerName("testedalgorithm");
			mlplan.call();
			LOGGER.info("ML-Plan execution completed.");

			/* evaluate solution produced by mlplan */
			SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
			ILearnerRunReport report = executor.execute(mlplan.getSelectedClassifier(), split.get(1));
			List<IPredictionAndGroundTruthTable<? extends Integer, ? extends ISingleLabelClassification>> l = Arrays.asList(report).stream()
					.map(r -> (IPredictionAndGroundTruthTable<? extends Integer, ? extends ISingleLabelClassification>) r.getPredictionDiffList()).collect(Collectors.toList());
			double errorRate = EAggregatedClassifierMetric.MEAN_ERRORRATE.loss(l);
			LOGGER.info("Error Rate of the solution produced by ML-Plan: {}", errorRate);
			Map<String, Object> map = new HashMap<>();
			map.put("timeout", mlplan.getTimeout().seconds());
			map.put("openmlid", openMLId);
			map.put("seed", seed);
			map.put("error_rate", errorRate);
			try {
				dbAdapter.insert(resultTable, map);
				LOGGER.info("Execution completed. Shutting down.");
				System.exit(0);
			} catch (Exception e) {
				LOGGER.error("Observed exception when trying to write result: {}", LoggerUtil.getExceptionInfo(e));
			}
		}
	}

}
