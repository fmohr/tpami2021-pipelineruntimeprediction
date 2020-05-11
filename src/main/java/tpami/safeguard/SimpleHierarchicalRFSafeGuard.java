package tpami.safeguard;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionOneLayerPartition;
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
			this.inferenceCalibrationFactor = 1.0;
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
		/*  Build component predictors */
		if (CONFIG.getBuildBaseComponents()) {
			// Load data...
			KVStoreCollection baseDefaultCol = DataBasedComponentPredictorUtil.readCSV(CONFIG.getBasicEvaluationRuntimeFile(), new HashMap<>());
			List<Integer> datasetIDs = new ArrayList<>(baseDefaultCol.stream().map(x -> x.getAsInt("openmlid")).collect(Collectors.toSet()));
			Collections.sort(datasetIDs);
			System.out.println(datasetIDs);

			baseDefaultCol.removeAnyContained(cleanTrainingData, true);
			rescaleApplicationTime(baseDefaultCol);

			Set<String> ids = baseDefaultCol.stream().map(x -> x.getAsString(CONFIG.getLabelForDatasetID())).collect(Collectors.toSet());
			List<String> idList = new ArrayList<>(ids);
			Collections.sort(idList);

			Set<String> components = baseDefaultCol.stream().map(x -> x.getAsString(CONFIG.getLabelForAlgorithm())).collect(Collectors.toSet());
			// Partition kvstore collections and fill predictor map
			KVStoreCollectionOneLayerPartition defaultPartition = new KVStoreCollectionOneLayerPartition(CONFIG.getLabelForAlgorithm(), baseDefaultCol);

			components.stream().map(component -> new Runnable() {
				@Override
				public void run() {
					// Speed up of build phase for base components (just for testing purposes)
					if (CONFIG.debuggingTestPipelineOnly() && !CONFIG.debuggingTestPipelineComponents().contains(component)) {
						return;
					}
					File paramCSV = new File(CONFIG.getBasicParameterizedDirectory(), String.format(CONFIG.getParameterizedFileNameTemplate(), DataBasedComponentPredictorUtil.mapWeka2ID(component)));
					KVStoreCollection paramCol = null;
					if (paramCSV.exists()) {
						try {
							paramCol = DataBasedComponentPredictorUtil.readCSV(paramCSV, new HashMap<>());
							paramCol.removeAnyContained(cleanTrainingData, true);
							rescaleApplicationTime(paramCol);
						} catch (IOException e) {
							LOGGER.warn("Could not read csv file {}", paramCSV.getAbsolutePath(), e);
						}
					}
					IBaseComponentEvaluationTimePredictor pred = null;
					try {
						pred = new BaseComponentEvaluationTimePredictor(component, defaultPartition.getData().get(component), paramCol);
					} catch (Exception e) {
						e.printStackTrace();
					}

					SimpleHierarchicalRFSafeGuard.this.lock.lock();
					try {
						SimpleHierarchicalRFSafeGuard.this.componentRuntimePredictorMap.put(component, pred);
					} finally {
						SimpleHierarchicalRFSafeGuard.this.lock.unlock();
					}
				}
			}).forEach(runnables::add);
		}

		// Build preprocessor data transformation predictors
		if (CONFIG.getBuildPreprocessorEffects()) {
			KVStoreCollection preprocessorData = DataBasedComponentPredictorUtil.readCSV(new File("python/data/metafeaturetransformations_essential.csv"), new HashMap<>());
			preprocessorData.removeAnyContained(cleanTrainingData, true);
			KVStoreCollectionOneLayerPartition preprocessorPartition = new KVStoreCollectionOneLayerPartition(CONFIG.getLabelForAlgorithm(), preprocessorData);

			preprocessorPartition.getData().entrySet().stream().map(preprocessorEntry -> new Runnable() {
				@Override
				public void run() {
					if (CONFIG.debuggingTestPipelineOnly() && !CONFIG.debuggingTestPipelineComponents().contains(preprocessorEntry.getKey())) {
						return;
					}

					KVStoreCollection parameterizedData = null;
					try {
						parameterizedData = DataBasedComponentPredictorUtil.readCSV(new File("python/data/parameterized/runtimes_" + preprocessorEntry.getKey() + "_parametrized.csv"), new HashMap<>());
					} catch (IOException e1) {
						e1.printStackTrace();
					}
					PreprocessingEffectPredictor pred;
					try {
						pred = new PreprocessingEffectPredictor(preprocessorEntry.getKey(), preprocessorEntry.getValue(), parameterizedData);
						SimpleHierarchicalRFSafeGuard.this.lock.lock();
						try {
							SimpleHierarchicalRFSafeGuard.this.preprocessingEffectPredictorMap.put(preprocessorEntry.getKey(), pred);
						} finally {
							SimpleHierarchicalRFSafeGuard.this.lock.unlock();
						}
					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			}).forEach(runnables::add);
		}

		// Build meta learner predictors
		if (CONFIG.getBuildMetaLearnerComponents()) {
			for (File csvFile : CONFIG.getMetaLearnerDirectory().listFiles()) {
				if (!csvFile.getName().endsWith(".csv")) {
					continue;
				}

				KVStoreCollection metaLearnerData = DataBasedComponentPredictorUtil.readCSV(csvFile, new HashMap<>());
				metaLearnerData.removeAnyContained(cleanTrainingData, true);
				String algorithm = metaLearnerData.get(0).getAsString(CONFIG.getLabelForAlgorithm());

				if (CONFIG.debuggingTestPipelineOnly() && !CONFIG.debuggingTestPipelineComponents().contains(algorithm)) {
					continue;
				}
				runnables.add(new Runnable() {
					@Override
					public void run() {
						try {
							MetaLearnerEvaluationTimePredictor pred = new MetaLearnerEvaluationTimePredictor(algorithm, metaLearnerData);
							SimpleHierarchicalRFSafeGuard.this.lock.lock();
							try {
								SimpleHierarchicalRFSafeGuard.this.metaLearnerPredictorMap.put(algorithm, pred);
							} finally {
								SimpleHierarchicalRFSafeGuard.this.lock.unlock();
							}
						} catch (Exception e) {
							LOGGER.error("Could not instantiate predictor for meta learner {}.", algorithm);
						}
					}
				});
			}
			LOGGER.info("Built meta learner evaluation time predictors:");
			this.metaLearnerPredictorMap.values().stream().map(x -> x.toString()).forEach(LOGGER::info);
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
			IMetaFeatureTransformationPredictor ppPred = this.preprocessingEffectPredictorMap.get(preprocessorID);
			MetaFeatureContainer metaFeaturesAfterPP = ppPred.predictTransformedMetaFeatures(preprocessor, metaFeaturesTrain);
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

			IMetaFeatureTransformationPredictor ppEffectPred = this.preprocessingEffectPredictorMap.get(DataBasedComponentPredictorUtil.componentInstanceToPreprocessorID(preprocessor));
			MetaFeatureContainer metaFeaturesTrainAfterPP = ppEffectPred.predictTransformedMetaFeatures(preprocessor, metaFeaturesTrain);
			MetaFeatureContainer metaFeaturesTestAfterPP = ppEffectPred.predictTransformedMetaFeatures(preprocessor, metaFeaturesTest);

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
