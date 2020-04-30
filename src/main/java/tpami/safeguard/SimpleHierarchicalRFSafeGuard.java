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
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.datastructure.kvstore.IKVStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionOneLayerPartition;
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
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;

public class SimpleHierarchicalRFSafeGuard implements IEvaluationSafeGuard {

	private static final boolean CONF_BUILD_BASE_COMPONENTS = true;
	private static final boolean CONF_BUILD_PREPROCESSOR_EFFECTS = true;
	private static final boolean CONF_BUILD_META_LEARNERS = true;

	private static final boolean CONF_TEST_PIPELINE_ONLY = true;
	private static final List<String> CONF_ACTIVE_PIPELINE_COMPONENTS = Arrays.asList("bf/cfssubseteval", J48.class.getName(), AdaBoostM1.class.getName());

	private static final File PARAMETERIZED_DIRECTORY = new File("python/data/parameterized/");
	private static final File META_LEARNER_DIR = new File("python/data/metalearner/");
	private static final String PARAMETERIZED_FILE_NAME_TEMPLATE = "runtimes_%s_parametrized_nooutliers.csv";

	private static final Logger LOGGER = LoggerFactory.getLogger(SimpleHierarchicalRFSafeGuard.class);

	private Lock lock = new ReentrantLock();

	private Map<String, IBaseComponentEvaluationTimePredictor> componentRuntimePredictorMap = new HashMap<>();
	private Map<String, IMetaFeatureTransformationPredictor> preprocessingEffectPredictorMap = new HashMap<>();
	private Map<String, IMetaLearnerEvaluationTimePredictor> metaLearnerPredictorMap = new HashMap<>();

	private static void rescaleApplicationTime(final KVStoreCollection col) {
		for (IKVStore store : col) {
			try {
				if (!store.getAsString("applicationtime").trim().isEmpty()) {
					store.put("applicationtime", store.getAsDouble("applicationtime") / store.getAsDouble("applicationsize") * BaseComponentEvaluationTimePredictor.SCALE_FOR_NUM_PREDICTIONS);
				}
			} catch (Exception e) {
				System.out.println(store);
				System.exit(0);
			}
		}
	}

	public SimpleHierarchicalRFSafeGuard(final File baseDefaultData, final int... excludeOpenMLDatasets) throws Exception {
		// Clean from excluded openml ids
		Map<String, Collection<String>> cleanTrainingData = new HashMap<>();
		cleanTrainingData.put("openmlid", Arrays.stream(excludeOpenMLDatasets).mapToObj(x -> x + "").collect(Collectors.toList()));

		/*  Build component predictors */
		if (CONF_BUILD_BASE_COMPONENTS) {
			// Load data...
			KVStoreCollection baseDefaultCol = DataBasedComponentPredictorUtil.readCSV(baseDefaultData, new HashMap<>());
			baseDefaultCol.removeAnyContained(cleanTrainingData, true);
			rescaleApplicationTime(baseDefaultCol);

			Set<String> ids = baseDefaultCol.stream().map(x -> x.getAsString("openmlid")).collect(Collectors.toSet());
			List<String> idList = new ArrayList<>(ids);
			Collections.sort(idList);

			Set<String> components = baseDefaultCol.stream().map(x -> x.getAsString("algorithm")).collect(Collectors.toSet());
			// Partition kvstore collections and fill predictor map
			KVStoreCollectionOneLayerPartition defaultPartition = new KVStoreCollectionOneLayerPartition("algorithm", baseDefaultCol);
			components.parallelStream().forEach(component -> {
				// Speed up of build phase for base components (just for testing purposes)
				if (CONF_TEST_PIPELINE_ONLY && !CONF_ACTIVE_PIPELINE_COMPONENTS.contains(component)) {
					return;
				}
				File paramCSV = new File(PARAMETERIZED_DIRECTORY, String.format(PARAMETERIZED_FILE_NAME_TEMPLATE, DataBasedComponentPredictorUtil.mapWeka2ID(component)));
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

				this.lock.lock();
				try {
					this.componentRuntimePredictorMap.put(component, pred);
				} finally {
					this.lock.unlock();
				}
			});

			LOGGER.info("Built base components successfully:");
			this.componentRuntimePredictorMap.values().stream().map(x -> x.toString()).forEach(LOGGER::info);
		}

		// Build preprocessor data transformation predictors
		if (CONF_BUILD_PREPROCESSOR_EFFECTS) {
			KVStoreCollection preprocessorData = DataBasedComponentPredictorUtil.readCSV(new File("python/data/metafeaturetransformations_essential.csv"), new HashMap<>());
			preprocessorData.removeAnyContained(cleanTrainingData, true);
			KVStoreCollectionOneLayerPartition preprocessorPartition = new KVStoreCollectionOneLayerPartition("algorithm", preprocessorData);

			preprocessorPartition.getData().entrySet().parallelStream().forEach(preprocessorEntry -> {
				if (CONF_TEST_PIPELINE_ONLY && !CONF_ACTIVE_PIPELINE_COMPONENTS.contains(preprocessorEntry.getKey())) {
					return;
				}

				// TODO: Load the data of the parameterized preprocessor.
				KVStoreCollection parameterizedData = null;
				PreprocessingEffectPredictor pred;
				try {
					pred = new PreprocessingEffectPredictor(preprocessorEntry.getKey(), preprocessorEntry.getValue(), parameterizedData);
					this.lock.lock();
					try {
						this.preprocessingEffectPredictorMap.put(preprocessorEntry.getKey(), pred);
					} finally {
						this.lock.unlock();
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
			LOGGER.info("Built preprocessor meta feature transformation effect predictors:");
			this.preprocessingEffectPredictorMap.values().stream().map(x -> x.toString()).forEach(LOGGER::info);
		}

		// Build meta learner predictors
		if (CONF_BUILD_META_LEARNERS) {
			for (File csvFile : META_LEARNER_DIR.listFiles()) {
				if (!csvFile.getName().endsWith(".csv")) {
					continue;
				}
				KVStoreCollection metaLearnerData = DataBasedComponentPredictorUtil.readCSV(csvFile, new HashMap<>());
				metaLearnerData.removeAnyContained(cleanTrainingData, true);
				String algorithm = metaLearnerData.get(0).getAsString("algorithm");

				if (CONF_TEST_PIPELINE_ONLY && !CONF_ACTIVE_PIPELINE_COMPONENTS.contains(algorithm)) {
					continue;
				}

				this.metaLearnerPredictorMap.put(algorithm, new MetaLearnerEvaluationTimePredictor(algorithm, metaLearnerData));
			}
			LOGGER.info("Built meta learner evaluation time predictors:");
			this.metaLearnerPredictorMap.values().stream().map(x -> x.toString()).forEach(LOGGER::info);
		}
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
		return this.predictInductionTime(ciw, new MetaFeatureContainer(dTrain));
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
		return this.predictInferenceTime(ciw, new MetaFeatureContainer(dTrain), new MetaFeatureContainer(dTest));
	}

	@Override
	public void updateWithActualInformation(final ComponentInstance ci, final ILabeledDataset<?> dTrain, final ILabeledDataset<?> dTest, final double inductionTime, final double inferenceTime) {
		this.updateWithActualInformation(ci, new MetaFeatureContainer(dTrain), new MetaFeatureContainer(dTest), inductionTime, inferenceTime);
	}

	public void updateWithActualInformation(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain, final MetaFeatureContainer metaFeaturesTest, final double inductionTime, final double inferenceTime) {
		MLComponentInstanceWrapper ciw = new MLComponentInstanceWrapper(ci);
		if (ciw.isBaseLearner()) {
			IBaseComponentEvaluationTimePredictor pred = this.componentRuntimePredictorMap.get(ciw.getComponent().getName());
			pred.setActualDefaultConfigurationInductionTime(metaFeaturesTrain, inductionTime);
			pred.setActualDefaultConfigurationInferenceTime(metaFeaturesTrain, metaFeaturesTest, inferenceTime);
		}
	}
}
