package tpami.safeguard;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.ISupervisedLearnerEvaluator;
import org.api4.java.algorithm.Timeout;
import org.api4.java.datastructure.kvstore.IKVStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.core.evaluation.evaluator.MonteCarloCrossValidationEvaluator;
import ai.libs.mlplan.core.ITimeTrackingLearner;
import ai.libs.mlplan.safeguard.IEvaluationSafeGuard;
import tpami.safeguard.api.EMetaFeature;
import tpami.safeguard.api.IBaseComponentEvaluationTimePredictor;
import tpami.safeguard.api.IMetaFeatureTransformationPredictor;
import tpami.safeguard.api.IMetaLearnerEvaluationTimePredictor;
import tpami.safeguard.impl.BaseComponentEvaluationTimePredictor;
import tpami.safeguard.impl.MetaFeatureContainer;
import tpami.safeguard.impl.MetaLearnerEvaluationTimePredictor;
import tpami.safeguard.impl.PreprocessingEffectPredictor;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import tpami.safeguard.util.MLComponentInstanceWrapper;

public class SimpleHierarchicalRFSafeGuard implements IEvaluationSafeGuard {

	private static final Logger LOGGER = LoggerFactory.getLogger(SimpleHierarchicalRFSafeGuard.class);
	private static final ISimpleHierarchicalRFSafeGuardConfig CONFIG = ConfigFactory.create(ISimpleHierarchicalRFSafeGuardConfig.class);

	public static final String FILE_PATTERN_BASIC_DEF = "runtimes_%s_default.csv";
	public static final String FILE_PATTERN_BASIC_PAR = "runtimes_%s_parametrized.csv";
	public static final String FILE_PATTERN_PREPROCESSOR = "%s.csv";
	public static final String FILE_PATTERN_METALEARNER = "metalearner_parametereffects_%s.csv";

	private Lock lock = new ReentrantLock();

	private Map<String, IBaseComponentEvaluationTimePredictor> componentRuntimePredictorMap = new HashMap<>();
	private Map<String, IMetaFeatureTransformationPredictor> preprocessingEffectPredictorMap = new HashMap<>();
	private Map<String, IMetaLearnerEvaluationTimePredictor> metaLearnerPredictorMap = new HashMap<>();

	private final ISupervisedLearnerEvaluator<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>> benchmark;
	private final ILabeledDataset<?> train;
	private final ILabeledDataset<?> test;
	private final double inductionCalibrationFactor;
	private final double inferenceCalibrationFactor;
	private final double benchmarkFactor;

	public static void setNumCPUs(final int numCPUs) {
		CONFIG.setProperty(ISimpleHierarchicalRFSafeGuardConfig.K_CPUS, numCPUs + "");
	}

	private static void rescaleApplicationTime(final KVStoreCollection col) {
		for (IKVStore store : col) {
			try {
				if (!store.getAsString(CONFIG.getLabelForApplicationTime()).trim().isEmpty()) {
					store.put(CONFIG.getLabelForApplicationTime(),
							store.getAsDouble(CONFIG.getLabelForApplicationTime()) / store.getAsDouble(CONFIG.getLabelForApplicationSize()) * BaseComponentEvaluationTimePredictor.SCALE_FOR_NUM_PREDICTIONS);
				}
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println(store);
				System.exit(0);
			}
		}
	}

	public SimpleHierarchicalRFSafeGuard(final int[] excludeOpenMLDatasets, final ISupervisedLearnerEvaluator<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>> benchmark, final ILabeledDataset<?> train,
			final ILabeledDataset<?> test) throws Exception {
		this.benchmark = benchmark;
		this.train = train;
		this.test = test;

		// Start calibration
		if (CONFIG.getPerformCalibration()) {
			Pair<Double, Double> calibrationFactors = new EvaluationTimeCalibrationModule(Arrays.stream(excludeOpenMLDatasets).mapToObj(x -> x + "").collect(Collectors.toList())).getSystemCalibrationFactor();
			this.inductionCalibrationFactor = calibrationFactors.getX();
			this.inferenceCalibrationFactor = calibrationFactors.getY();
		} else {
			this.inductionCalibrationFactor = 1.0;
			this.inferenceCalibrationFactor = 0.05;
		}
		System.err.println("Calibration factors: " + this.inductionCalibrationFactor + " / " + this.inferenceCalibrationFactor);

		// Extract benchmark factor
		if (benchmark instanceof MonteCarloCrossValidationEvaluator) {
			this.benchmarkFactor = ((MonteCarloCrossValidationEvaluator) benchmark).getRepeats();
		} else {
			this.benchmarkFactor = 1.0;
		}
		System.err.println("Benchmark factor: " + this.benchmarkFactor);

		// Clean from excluded openml ids
		Map<String, Collection<String>> cleanTrainingData = new HashMap<>();
		cleanTrainingData.put(CONFIG.getLabelForDatasetID(), Arrays.stream(excludeOpenMLDatasets).mapToObj(x -> x + "").collect(Collectors.toList()));

		List<Runnable> runnables = new ArrayList<>();
		// Build component predictors
		if (CONFIG.getBuildBaseComponents()) {
			// Build runtime predictors for basic components (base learners + preprocessors)
			CONFIG.getBasicComponentsForRuntime().stream().forEach(name -> {
				// Speed up of build phase for base components (just for testing purposes)
				if (CONFIG.debuggingTestPipelineOnly() && !CONFIG.debuggingTestPipelineComponents().contains(name)) {
					LOGGER.info("Skip building base component model for " + name);
					return;
				}

				runnables.add(new Runnable() {
					@Override
					public void run() {

						try {
							// read dataset for default parameterization
							File defaultDatasetFile = new File(CONFIG.getBasicComponentsForDefaultRuntimeDirectory(), String.format(FILE_PATTERN_BASIC_DEF, name));
							long start = System.currentTimeMillis();
							KVStoreCollection defaultCol = DataBasedComponentPredictorUtil.readCSV(defaultDatasetFile, new HashMap<>());
							System.out.println("File read for " + defaultDatasetFile + " required " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
							// read dataset for parameterized data points
							File paramDatasetFile = new File(CONFIG.getBasicComponentsForDefaultRuntimeDirectory(), String.format(FILE_PATTERN_BASIC_PAR, name));
							KVStoreCollection paramCol = null;
							if (paramDatasetFile.exists()) {
								start = System.currentTimeMillis();
								paramCol = DataBasedComponentPredictorUtil.readCSV(paramDatasetFile, new HashMap<>());
								System.out.println("File read of " + paramDatasetFile + " required " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
							}

							// clean from excluded dataset ids
							// rescale application times
							defaultCol.removeAnyContained(cleanTrainingData, true);
							rescaleApplicationTime(defaultCol);
							if (paramCol != null) {
								paramCol.removeAnyContained(cleanTrainingData, true);
								rescaleApplicationTime(paramCol);
							}

							// Build predictor for this component
							IBaseComponentEvaluationTimePredictor pred = null;
							try {
								pred = new BaseComponentEvaluationTimePredictor(name, defaultCol, paramCol);
							} catch (Exception e) {
								e.printStackTrace();
							}

							SimpleHierarchicalRFSafeGuard.this.lock.lock();
							try {
								SimpleHierarchicalRFSafeGuard.this.componentRuntimePredictorMap.put(DataBasedComponentPredictorUtil.mapID2Weka(name), pred);
							} finally {
								SimpleHierarchicalRFSafeGuard.this.lock.unlock();
							}
							LOGGER.info("Done building base component model for " + name);
						} catch (Exception e) {
							LOGGER.warn("Could not build predictor for {} due to exception", name, e);
						}
					}
				});
			});
		}

		// Build preprocessor data transformation predictors
		if (CONFIG.getBuildPreprocessorEffects()) {
			CONFIG.getPreprocessorsForTransformEffect().stream().forEach(name -> {
				// Speed up of build phase for preprocessing effects (just for testing purposes)
				if (CONFIG.debuggingTestPipelineOnly() && !CONFIG.debuggingTestPipelineComponents().contains(name)) {
					LOGGER.info("Skip building preprocessor transform effect model for " + name);
					return;
				}
				runnables.add(new Runnable() {
					@Override
					public void run() {

						try {
							// Read in dataset for learning predictors of meta features transformation by preprocessors.
							File ppTransformFile = new File(CONFIG.getPreprocessorsForTransformEffectDirectory(), String.format(FILE_PATTERN_PREPROCESSOR, name));
							long start = System.currentTimeMillis();
							KVStoreCollection ppTransformData = DataBasedComponentPredictorUtil.readCSV(ppTransformFile, new HashMap<>());
							System.out.println("File read of " + ppTransformFile + " required " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");

							// Build predictor
							PreprocessingEffectPredictor pred = new PreprocessingEffectPredictor(name, ppTransformData);
							SimpleHierarchicalRFSafeGuard.this.lock.lock();
							try {
								SimpleHierarchicalRFSafeGuard.this.preprocessingEffectPredictorMap.put(name, pred);
							} finally {
								SimpleHierarchicalRFSafeGuard.this.lock.unlock();
							}
							LOGGER.info("Done building preprocessor transform effect model for " + name);
						} catch (Exception e) {
							LOGGER.warn("Could not build preprocessing transform predictor for {} due to exception", name, e);
						}
					}
				});
			});
		}

		// Build meta learner predictors
		if (CONFIG.getBuildMetaLearnerComponents()) {
			CONFIG.getMetaLearnerTransformEffect().stream().forEach(name -> {
				// Speed up of build phase for meta learner effects (just for testing purposes)
				if (CONFIG.debuggingTestPipelineOnly() && !CONFIG.debuggingTestPipelineComponents().contains(name)) {
					LOGGER.info("Skip building meta learner transform effect model for " + name);
					return;
				}

				runnables.add(new Runnable() {
					@Override
					public void run() {

						try {
							// Read in dataset for fitting predictors of meta features transformation by meta learners.
							File metaLearnerFile = new File(CONFIG.getMetaLearnerTransformEffectDirectory(), String.format(FILE_PATTERN_METALEARNER, name));
							long start = System.currentTimeMillis();
							KVStoreCollection metaLearnerData = DataBasedComponentPredictorUtil.readCSV(metaLearnerFile, new HashMap<>());
							System.out.println("File read of " + metaLearnerFile + " required " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");

							// Build predictor
							MetaLearnerEvaluationTimePredictor pred = new MetaLearnerEvaluationTimePredictor(name, metaLearnerData);
							SimpleHierarchicalRFSafeGuard.this.lock.lock();
							try {
								SimpleHierarchicalRFSafeGuard.this.metaLearnerPredictorMap.put(DataBasedComponentPredictorUtil.mapID2Weka(name), pred);
							} finally {
								SimpleHierarchicalRFSafeGuard.this.lock.unlock();
							}
							LOGGER.info("Done building meta leaner transform effect model for " + name);
						} catch (Exception e) {
							LOGGER.warn("Could not instantiate predictor for meta learner {}.", name);
						}
					}
				});
			});
		}

		ExecutorService pool = Executors.newFixedThreadPool(CONFIG.getNumCPUs());
		runnables.stream().forEach(pool::submit);

		pool.shutdown();
		pool.awaitTermination(1, TimeUnit.HOURS);

		LOGGER.info("Instantiated safe guard with basic component predictors for:\n{}", this.componentRuntimePredictorMap.keySet().stream().collect(Collectors.joining("\n")));
		LOGGER.info("Instantiated safe guard with preprocessing effect predictors for:\n{}", this.preprocessingEffectPredictorMap.keySet().stream().collect(Collectors.joining("\n")));
		LOGGER.info("Instantiated safe guard with meta learning predictors for:\n{}", this.metaLearnerPredictorMap.keySet().stream().collect(Collectors.joining("\n")));
	}

	@Override
	public boolean predictWillAdhereToTimeout(final ComponentInstance ci, final Timeout timeout) throws Exception {
		return this.predictEvaluationTime(ci, this.train, this.test) * this.benchmarkFactor < timeout.seconds();
	}

	public double predictInductionTime(final MLComponentInstanceWrapper ciw, final MetaFeatureContainer metaFeaturesTrain) throws Exception {
		if (ciw.isPipeline()) {
			LOGGER.debug("We have a pipeline here: {}", ciw.getComponent().getName());
			double inductionTime = 0.0;
			// Compute runtime for executing preprocessor
			MLComponentInstanceWrapper preprocessor = ciw.getPreprocessor();
			inductionTime += this.predictInductionTime(preprocessor, metaFeaturesTrain);
			LOGGER.debug("Runtime after inducing preprocessor: {}", inductionTime);
			inductionTime += this.predictInferenceTime(preprocessor, metaFeaturesTrain, metaFeaturesTrain);
			LOGGER.debug("Runtime after applying preprocessor: {}", inductionTime);
			// Compute features for after
			String preprocessorID = DataBasedComponentPredictorUtil.componentInstanceToPreprocessorID(preprocessor);
			MetaFeatureContainer metaFeaturesAfterPP = new MetaFeatureContainer(metaFeaturesTrain);

			if (this.preprocessingEffectPredictorMap.containsKey(preprocessorID)) {
				IMetaFeatureTransformationPredictor ppPred = this.preprocessingEffectPredictorMap.get(preprocessorID);
				metaFeaturesAfterPP = ppPred.predictTransformedMetaFeatures(preprocessor, metaFeaturesTrain);
			}
			LOGGER.debug("Meta features after preprocessing: {}", metaFeaturesAfterPP);

			// Compute runtime for classifier
			inductionTime += this.predictInductionTime(ciw.getClassifier(), metaFeaturesAfterPP);
			LOGGER.debug("Runtime after inducing classifier: {}", inductionTime);

			return inductionTime;
		} else if (ciw.isPreprocessor()) {
			String preprocessorIdentifier = DataBasedComponentPredictorUtil.componentInstanceToPreprocessorID(ciw);
			if (preprocessorIdentifier != null && this.componentRuntimePredictorMap.containsKey(preprocessorIdentifier)) {
				return this.componentRuntimePredictorMap.get(preprocessorIdentifier).predictInductionTime(ciw, metaFeaturesTrain);
			} else {
				LOGGER.warn("Unknown component instance for preprocessor: {}", ciw);
				return 0.0;
			}
		} else if (ciw.isMetaLearner()) {
			IMetaLearnerEvaluationTimePredictor mlPred = this.metaLearnerPredictorMap.get(ciw.getComponent().getName());
			if (mlPred != null) {
				return mlPred.predictInductionTime(ciw, this.componentRuntimePredictorMap.get(ciw.getBaseLearner().getComponent().getName()), metaFeaturesTrain);
			} else {
				LOGGER.warn("I do not know this meta learner, so I predict a runtime of 0 for {}", ciw);
				return 0.0;
			}
		} else if (ciw.isBaseLearner()) {
			return this.componentRuntimePredictorMap.get(ciw.getComponent().getName()).predictInductionTime(ciw, metaFeaturesTrain);
		} else {
			LOGGER.warn("Case not covered. This component instance {} seems to be neither pipeline nor meta learner nor base learner. Therefore we return 0 as time which might be overly enthusiastic.", ciw);
			return 0.0;
		}
	}

	@Override
	public double predictInductionTime(final ComponentInstance ci, final ILabeledDataset<?> dTrain) throws Exception {
		MLComponentInstanceWrapper ciw;
		if (ci instanceof MLComponentInstanceWrapper) {
			ciw = (MLComponentInstanceWrapper) ci;
		} else {
			ciw = new MLComponentInstanceWrapper(ci);
		}
		return this.predictInductionTime(ciw, new MetaFeatureContainer(dTrain)) * this.inductionCalibrationFactor;
	}

	public double predictInferenceTime(final MLComponentInstanceWrapper ciw, final MetaFeatureContainer metaFeaturesTrain, final MetaFeatureContainer metaFeaturesTest) throws Exception {
		if (ciw.isPipeline()) {
			double inferenceTime = 0.0;
			MLComponentInstanceWrapper preprocessor = ciw.getPreprocessor();
			inferenceTime += this.predictInferenceTime(preprocessor, metaFeaturesTrain, metaFeaturesTest);
			LOGGER.debug("Runtime after applying preprocessor to test data: {}", inferenceTime);

			String ppID = DataBasedComponentPredictorUtil.componentInstanceToPreprocessorID(preprocessor);

			MetaFeatureContainer metaFeaturesTrainAfterPP = new MetaFeatureContainer(metaFeaturesTrain);
			MetaFeatureContainer metaFeaturesTestAfterPP = new MetaFeatureContainer(metaFeaturesTest);

			if (this.preprocessingEffectPredictorMap.containsKey(ppID)) {
				IMetaFeatureTransformationPredictor ppEffectPred = this.preprocessingEffectPredictorMap.get(ppID);
				metaFeaturesTrainAfterPP = ppEffectPred.predictTransformedMetaFeatures(preprocessor, metaFeaturesTrain);
				metaFeaturesTestAfterPP = ppEffectPred.predictTransformedMetaFeatures(preprocessor, metaFeaturesTest);
			}

			inferenceTime += this.predictInferenceTime(ciw.getClassifier(), metaFeaturesTrainAfterPP, metaFeaturesTestAfterPP);
			LOGGER.debug("Runtime after applying classifier to test data: {}", inferenceTime);
			return inferenceTime;
		} else if (ciw.isPreprocessor()) {
			String preprocessorIdentifier = DataBasedComponentPredictorUtil.componentInstanceToPreprocessorID(ciw);
			if (preprocessorIdentifier != null && this.componentRuntimePredictorMap.containsKey(preprocessorIdentifier)) {
				return this.componentRuntimePredictorMap.get(preprocessorIdentifier).predictInferenceTime(ciw, metaFeaturesTrain) / IBaseComponentEvaluationTimePredictor.SCALE_FOR_NUM_PREDICTIONS
						* metaFeaturesTest.getFeature(EMetaFeature.NUM_INSTANCES);
			} else {
				LOGGER.warn("Unknown component instance for preprocessor: {}", ciw);
				return 0.0;
			}
		} else if (ciw.isMetaLearner()) {
			IMetaLearnerEvaluationTimePredictor mlPred = this.metaLearnerPredictorMap.get(ciw.getComponent().getName());
			if (mlPred != null) {
				return mlPred.predictInferenceTime(ciw, this.componentRuntimePredictorMap.get(ciw.getBaseLearner().getComponent().getName()), metaFeaturesTrain, metaFeaturesTest);
			} else {
				LOGGER.warn("I do not know this meta learner, so I predict a runtime of 0 for {}", ciw);
				return 0.0;
			}
		} else if (ciw.isBaseLearner()) {
			return this.componentRuntimePredictorMap.get(ciw.getComponent().getName()).predictInferenceTime(ciw, metaFeaturesTrain, metaFeaturesTest);
		} else {
			LOGGER.warn("Case not covered. This component instance {} seems to be neither pipeline nor meta learner nor base learner. Therefore we return 0 as time which might be overly enthusiastic.", ciw);
			return 0.0;
		}

	}

	@Override
	public double predictInferenceTime(final ComponentInstance ci, final ILabeledDataset<?> dTrain, final ILabeledDataset<?> dTest) throws Exception {
		MLComponentInstanceWrapper ciw;
		if (ci instanceof MLComponentInstanceWrapper) {
			ciw = (MLComponentInstanceWrapper) ci;
		} else {
			ciw = new MLComponentInstanceWrapper(ci);
		}
		return this.predictInferenceTime(ciw, new MetaFeatureContainer(dTrain), new MetaFeatureContainer(dTest)) * this.inferenceCalibrationFactor;
	}

	@Override
	public void updateWithActualInformation(final ComponentInstance ci, final ITimeTrackingLearner wrappedLearner) {
		MLComponentInstanceWrapper ciw = new MLComponentInstanceWrapper(ci);
		if (ciw.isBaseLearner()) {
			IBaseComponentEvaluationTimePredictor pred = this.componentRuntimePredictorMap.get(ciw.getComponent().getName());
			pred.setActualDefaultConfigurationInductionTime(new MetaFeatureContainer(this.train), wrappedLearner.getFitTimes().stream().mapToDouble(x -> (double) x).average().getAsDouble());
			pred.setActualDefaultConfigurationInferenceTime(new MetaFeatureContainer(this.train), new MetaFeatureContainer(this.test), wrappedLearner.getBatchPredictionTimes().stream().mapToDouble(x -> (double) x).average().getAsDouble());
		}
	}

}
