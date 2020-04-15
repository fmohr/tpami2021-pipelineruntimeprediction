package tpami.automlexperimenter;

import java.io.File;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.core.evaluation.execution.ISupervisedLearnerExecutor;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionFailedException;
import org.api4.java.ai.ml.core.evaluation.execution.LearnerExecutionInterruptedException;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.google.common.eventbus.Subscribe;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.evaluation.evaluator.events.TrainTestSplitEvaluationFailedEvent;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.mlplan.core.ITimeTrackingLearner;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.core.TimeTrackingLearnerWrapper;
import ai.libs.mlplan.core.events.TimeTrackingLearnerEvaluationEvent;
import ai.libs.mlplan.multiclass.wekamlplan.MLPlanWekaBuilder;
import ai.libs.mlplan.safeguard.AlwaysEvaluateSafeGuardFactory;
import ai.libs.mlplan.safeguard.AlwaysPreventSafeGuardFactory;
import ai.libs.mlplan.safeguard.EvaluationSafeGuardFiredEvent;
import ai.libs.mlplan.safeguard.IEvaluationSafeGuard;
import tpami.safeguard.CalibrationConstantsDeterminedEvent;
import tpami.safeguard.SimpleHierarchicalRFSafeGuardFactory;

public class AutoMLExperimenter {

	/**
	 * Variables for the experiment and database setup
	 */
	private static final File configFile = new File("automlexperimenter.properties");
	private static final IExampleMCCConfig m = (IExampleMCCConfig) ConfigCache.getOrCreate(IExampleMCCConfig.class).loadPropertiesFromFile(configFile);
	private static final IDatabaseConfig dbconfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(configFile);
	private static final IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(dbconfig);

	private static SQLAdapter adapter = new SQLAdapter(dbconfig);

	enum ESafeGuardType {
		ALWAYS_EVAL, ALWAYS_PREVENT, HIERARCHICAL;
	}

	private static final ESafeGuardType SAFEGUARD_TYPE = ESafeGuardType.HIERARCHICAL;

	public AutoMLExperimenter() {
		// TODO Auto-generated constructor stub
	}

	public static void main(final String[] args)
			throws AlgorithmTimeoutedException, ExperimentDBInteractionFailedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		if (args.length > 0) {
			switch (args[0]) {
			case "create":
				System.out.println("Create table with experiments.");
				createTableWithExperiments();
				break;
			case "run":
				System.out.println("Run experiments");
				runExperiments();
				break;
			}
		} else {
			System.out.println("Run experiments");
//			createTableWithExperiments();
			runExperiments();
		}
	}

	public static void createTableWithExperiments()
			throws ExperimentDBInteractionFailedException, AlgorithmTimeoutedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(m, dbHandle);
		preparer.synchronizeExperiments();
	}

	public static void deleteTable() throws ExperimentDBInteractionFailedException {
		dbHandle.deleteDatabase();
	}

	public static void runExperiments() throws ExperimentDBInteractionFailedException, InterruptedException {
		System.out.println("Create sql adapter for eval logs...");
		System.out.println("Create experiment runner...");
		ExperimentRunner runner = new ExperimentRunner(m, new IExperimentSetEvaluator() {
			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws InterruptedException, ExperimentEvaluationFailedException {
				System.out.println("Evaluate");

				Map<String, int[]> excludeIDs = new HashMap<>();
				excludeIDs.put("abalone", new int[] { 183, 720, 1557 });
				excludeIDs.put("amazon", new int[] { 1457 });
				excludeIDs.put("car", new int[] { 991, 40975 });
				excludeIDs.put("cifar10", new int[] { 41983, 40926, 40927 });
				excludeIDs.put("cifar10small", new int[] { 41983, 40926, 40927 });
				excludeIDs.put("convex", new int[] {}); // not available on openml
				excludeIDs.put("credit-g", new int[] { 31 });
				excludeIDs.put("dexter", new int[] { 4136 });
				excludeIDs.put("dorothea", new int[] { 4137 });
				excludeIDs.put("gisette", new int[] { 41026 });
				excludeIDs.put("krvskp", new int[] { 1481 });
				excludeIDs.put("madelon", new int[] { 1485 });
				excludeIDs.put("mnist", new int[] { 554 });
				excludeIDs.put("mnistrotationbackimagenew", new int[] { 41065 });
				excludeIDs.put("secom", new int[] {}); // not available on openml
				excludeIDs.put("semeion", new int[] { 1501, 41973 });
				excludeIDs.put("shuttle", new int[] { 40685 });
				excludeIDs.put("waveform", new int[] { 60, 979, 4551 });
				excludeIDs.put("winequality", new int[] { 42184 });
				excludeIDs.put("yeast", new int[] { 181, 316, 40597, 41091, 41473 });

				try {
					Map<String, String> description = experimentEntry.getExperiment().getValuesOfKeyFields();
					long seed = Long.parseLong(description.get("seed"));
					String datasetName = description.get("dataset");
					String algorithmMode = description.get("algorithmmode");
					int totalTimeoutInS = Integer.parseInt(description.get("timeout"));
					int evaluationTimeoutInS = Integer.parseInt(description.get("evaltimeout"));

					System.out.println("Executing experiment with ID " + experimentEntry.getId() + " with " + experimentEntry.getExperiment().getNumCPUs() + " CPUs / " + experimentEntry.getExperiment().getMemoryInMB() + "MB RAM"
							+ " and description " + description);

					File datasetFile = new File(m.getDatasetFolder(), datasetName + ".arff");
					ILabeledDataset<?> dataset = ArffDatasetAdapter.readDataset(datasetFile);
					List<ILabeledDataset<?>> trainTestSplit = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, seed, 0.7);

					/* get experiment setup */

					MLPlanWekaBuilder builder = new MLPlanWekaBuilder();
					builder.withNumCpus(experimentEntry.getExperiment().getNumCPUs());
					builder.withTimeOut(new Timeout(totalTimeoutInS, TimeUnit.SECONDS));
					builder.withNodeEvaluationTimeOut(new Timeout(evaluationTimeoutInS * 3, TimeUnit.SECONDS));
					builder.withCandidateEvaluationTimeOut(new Timeout(evaluationTimeoutInS, TimeUnit.SECONDS));
					builder.withDataset(trainTestSplit.get(0));

					if (algorithmMode.equals("safeguard")) {
						switch (SAFEGUARD_TYPE) {
						case ALWAYS_EVAL:
							builder.withSafeGuardFactory(new AlwaysEvaluateSafeGuardFactory());
							break;
						case ALWAYS_PREVENT:
							builder.withSafeGuardFactory(new AlwaysPreventSafeGuardFactory());
							break;
						case HIERARCHICAL:
							SimpleHierarchicalRFSafeGuardFactory safeguardFactory = new SimpleHierarchicalRFSafeGuardFactory();
							safeguardFactory.setNumCPUs(experimentEntry.getExperiment().getNumCPUs());
							if (excludeIDs.containsKey(datasetName)) {
								safeguardFactory.withExcludeOpenMLDatasets(excludeIDs.get(datasetName));
							} else {
								System.err.println("Dataset with name " + datasetName + " not contained in the excludeIDs map.");
								safeguardFactory.withExcludeOpenMLDatasets(new int[] {});
							}

							safeguardFactory.build();
							builder.withSafeGuardFactory(safeguardFactory);
							break;
						}

					}
					/* create objects for experiment */
//					logger.info("Evaluate {} for dataset {} and seed {}", algorithmMode, datasetName, seed);

					MLPlan<IWekaClassifier> mlplan = builder.build();
					mlplan.setLoggerName("mlplan");
					mlplan.registerListener(new Object() {
						@Subscribe
						public void rcvClassifierFoundEvent(final TimeTrackingLearnerEvaluationEvent event) {
							this.logCandidateEvaluation("success", event.getComponentInstance(), event.getScore() + "", event.getActualFitTime(), event.getActualPredictTime(), event.getPredictedFitTime(), event.getPredictedPredictTime());
						}

						@Subscribe
						public void rcvTrainTestSplitEvaluationFailedEvent(final TrainTestSplitEvaluationFailedEvent<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>> event) {
							try {

								TimeTrackingLearnerWrapper learner = null;
								if (event.getLearner() instanceof ITimeTrackingLearner) {
									learner = (TimeTrackingLearnerWrapper) event.getLearner();
								}
								if (learner != null) {
									if (event.getReport().getException() instanceof LearnerExecutionInterruptedException) {
										LearnerExecutionInterruptedException e = ((LearnerExecutionInterruptedException) event.getReport().getException());

										Double actualFitTime = (double) (e.getTrainTimeEnd() - e.getTrainTimeStart()) / 1000;
										Double actualPredictTime = (double) (e.getTestTimeEnd() - e.getTestTimeStart()) / 1000;
										Double predictedFitTime = learner.getPredictedInductionTime();
										Double predictedPredictTime = learner.getPredictedInferenceTime();

										this.logCandidateEvaluation("timeout", learner.getComponentInstance(), event.getReport().toString(), actualFitTime, actualPredictTime, predictedFitTime, predictedPredictTime);
									} else if (event.getReport().getException() instanceof LearnerExecutionFailedException) {
										LearnerExecutionFailedException e = ((LearnerExecutionFailedException) event.getReport().getException());

										Double actualFitTime = (double) (e.getTrainTimeEnd() - e.getTrainTimeStart()) / 1000;
										Double actualPredictTime = (double) (e.getTestTimeEnd() - e.getTestTimeStart()) / 1000;
										Double predictedFitTime = learner.getPredictedInductionTime();
										Double predictedPredictTime = learner.getPredictedInferenceTime();

										this.logCandidateEvaluation("crashed", learner.getComponentInstance(), event.getReport().toString(), actualFitTime, actualPredictTime, predictedFitTime, predictedPredictTime);
									}
								} else {
									if (event.getReport().getException() instanceof LearnerExecutionInterruptedException) {
										LearnerExecutionInterruptedException e = ((LearnerExecutionInterruptedException) event.getReport().getException());

										this.logCandidateEvaluation("timeout", null, event.getReport().toString(), -1.0, -1.0, -1.0, -1.0);
									} else if (event.getReport().getException() instanceof LearnerExecutionFailedException) {
										LearnerExecutionFailedException e = ((LearnerExecutionFailedException) event.getReport().getException());

										this.logCandidateEvaluation("crashed", null, event.getReport().toString(), -1.0, -1.0, -1.0, -1.0);
									}

								}
							} catch (Throwable e) {
								e.printStackTrace();
							}
						}

						@Subscribe
						public void rcvCalibrationConstants(final CalibrationConstantsDeterminedEvent e) {
							Map<String, Object> constants = new HashMap<>();
							constants.put("c_induction", e.getCInduction());
							constants.put("c_inference", e.getCInference());
							processor.processResults(constants);
						}

						@Subscribe
						public void rcvEvaluationSafeGuardFiredEvent(final EvaluationSafeGuardFiredEvent e) {
							Double predictedFitTime = Double.parseDouble(e.getComponentInstance().getAnnotation(IEvaluationSafeGuard.ANNOTATION_PREDICTED_INDUCTION_TIME));
							Double predictedPredictTime = Double.parseDouble(e.getComponentInstance().getAnnotation(IEvaluationSafeGuard.ANNOTATION_PREDICTED_INFERENCE_TIME));
							this.logCandidateEvaluation("safeguard", e.getComponentInstance(), "Prevented execution", -1.0, -1.0, predictedFitTime, predictedPredictTime);
						}

						private void logCandidateEvaluation(final String status, final ComponentInstance ci, final String result, final Double actualFitTime, final Double actualPredictTime, final Double predictedFitTime,
								final Double predictedPredictTime) {
							try {
								Map<String, Object> map = new HashMap<>();
								map.put("status", status);
								map.put("component_instance", new ComponentInstanceAdapter().componentInstanceToString(ci));
								map.put("result", result);
								map.put("actualFitTime", actualFitTime);
								map.put("actualPredictTime", actualPredictTime);
								map.put("predictedFitTime", predictedFitTime);
								map.put("predictedPredictTime", predictedPredictTime);
								map.put("thread", Thread.currentThread().getName());
								map.put("experiment_id", experimentEntry.getId());
								map.put("timestamp_found", System.currentTimeMillis());

								new KVStore(map);
								System.out.println("INSERT " + m.getEvalTable() + " " + map);
								adapter.insert(m.getEvalTable(), map);
							} catch (JsonProcessingException e) {
								e.printStackTrace();
							} catch (SQLException e) {
								e.printStackTrace();
							}
						}

					});
					IWekaClassifier model = mlplan.call();

					ISupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
					ILearnerRunReport report = executor.execute(model, trainTestSplit.get(1));

					String candidate = new ComponentInstanceAdapter().componentInstanceToString(mlplan.getComponentInstanceOfSelectedClassifier());
					double loss = EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList());

					/* run fictive experiment */
					Map<String, Object> results = new HashMap<>();
					results.put("loss", loss);
					results.put("candidate", candidate);
					/* report results */
					processor.processResults(results);
				} catch (InterruptedException e) {
					throw e;
				} catch (Exception e) {
					throw new ExperimentEvaluationFailedException(e);
				}
			}
		}, dbHandle);
		runner.setCheckMemory(false);
		runner.randomlyConductExperiments(1);
	}
}
