package tpami.automlexperimenter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.core.evaluation.execution.ISupervisedLearnerExecutor;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;

import com.google.common.eventbus.Subscribe;

import ai.libs.jaicore.db.IDatabaseConfig;
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
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.multiclass.wekamlplan.MLPlanWekaBuilder;
import tpami.safeguard.SimpleHierarchicalRFSafeGuardFactory;

public class MLPlanExperimenter {

	/**
	 * Variables for the experiment and database setup
	 */
	private static final File configFile = new File("automlexperimenter.properties");
	private static final IExampleMCCConfig m = (IExampleMCCConfig) ConfigCache.getOrCreate(IExampleMCCConfig.class).loadPropertiesFromFile(configFile);
	private static final IDatabaseConfig dbconfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(configFile);
	private static final IExperimentDatabaseHandle dbHandle = new ExperimenterMySQLHandle(dbconfig);
//	private static final Logger logger = LoggerFactory.getLogger(MachineLearningExperimenter.class);

	public static void main(final String[] args)
			throws ExperimentDBInteractionFailedException, AlgorithmTimeoutedException, IllegalExperimentSetupException, ExperimentAlreadyExistsInDatabaseException, InterruptedException, AlgorithmExecutionCanceledException {
//		System.out.println(logger.getName());
		System.out.println("Hallo");
		System.out.println(Arrays.toString(args));
		if (args.length > 0) {
			switch (args[0]) {
			case "-c":
//				logger.info("Create table with experiments.");
				createTableWithExperiments();
				break;
			case "-r":
//				logger.info("Run experiments");
				runExperiments();
				break;
			}
		} else {
//			logger.info("Run experiments");
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
		ExperimentRunner runner = new ExperimentRunner(m, new IExperimentSetEvaluator() {
			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws InterruptedException, ExperimentEvaluationFailedException {
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

					File datasetFile = new File(m.getDatasetFolder(), datasetName + ".arff");
					ILabeledDataset<?> dataset = ArffDatasetAdapter.readDataset(datasetFile);
					List<ILabeledDataset<?>> trainTestSplit = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, seed, 0.7);

					/* get experiment setup */

					MLPlanWekaBuilder builder = new MLPlanWekaBuilder();
					builder.withTimeOut(new Timeout(totalTimeoutInS, TimeUnit.SECONDS));
					builder.withCandidateEvaluationTimeOut(new Timeout(evaluationTimeoutInS, TimeUnit.SECONDS));
					builder.withDataset(trainTestSplit.get(0));

					if (algorithmMode.equals("safeguard")) {
						SimpleHierarchicalRFSafeGuardFactory safeguardFactory = new SimpleHierarchicalRFSafeGuardFactory();
						if (excludeIDs.containsKey(datasetName)) {
							safeguardFactory.withExcludeOpenMLDatasets(excludeIDs.get(datasetName));
						} else {
							System.err.println("Dataset with name " + datasetName + " not contained in the excludeIDs map.");
							safeguardFactory.withExcludeOpenMLDatasets(new int[] {});
						}
						builder.withSafeGuardFactory(safeguardFactory);
					}
					/* create objects for experiment */
//					logger.info("Evaluate {} for dataset {} and seed {}", algorithmMode, datasetName, seed);

					MLPlan<IWekaClassifier> mlplan = builder.build();
					mlplan.registerListener(new Object() {
						@Subscribe
						public void rcvEvent(final Object event) {
							if (event.getClass().getName().contains("graphvisualizer")) {
								return;
							}

							try (BufferedWriter bw = new BufferedWriter(new FileWriter(new File("event.log"), true))) {
								bw.write(event.getClass().getName() + "\n");
							} catch (IOException e) {
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

					long timeStartTraining = System.currentTimeMillis();
					results.put("traintime", System.currentTimeMillis() - timeStartTraining);

					/* report results */
					results.put("loss", loss);
					processor.processResults(results);
				} catch (InterruptedException e) {
					throw e;
				} catch (Exception e) {
					throw new ExperimentEvaluationFailedException(e);
				}
			}
		}, dbHandle);
		runner.setCheckMemory(false);
		runner.randomlyConductExperiments(-1);
	}
}
