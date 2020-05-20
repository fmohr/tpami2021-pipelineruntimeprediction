package tpami.safeguard.impl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.mlplan.safeguard.IEvaluationSafeGuard;
import tpami.safeguard.api.EMetaFeature;
import tpami.safeguard.api.IBaseComponentEvaluationTimePredictor;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class BaseComponentEvaluationTimePredictor implements IBaseComponentEvaluationTimePredictor {

	private static final Logger LOGGER = LoggerFactory.getLogger(BaseComponentEvaluationTimePredictor.class);

	private static final String[] FEATURES_INDUCTION = { "fitsize", "numattributes" };
	private static final String TARGET_INDUCTION = "fittime";
	private static final String[] FEATURES_INFERENCE = { "fitsize", "numattributes" };
	private static final String TARGET_INFERENCE = "applicationtime";
	private static final List<String> NON_PARAMETER_COLUMNS = Arrays.asList("openmlid", "totalsize", "fitsize", "applicationsize", "seed", "algorithm", "algorithmoptions", "fittime", "applicationtime", "fittime_def", "applicationtime_def",
			"openmlid", "totalsize", "algorithm", "algorithmoptions", "seed", "fitsize", "numattributes", "numlabels", "numnumericattributes", "numsymbolicattributes", "numberofcategories", "numericattributesafterbinarization",
			"totalvariance", "attributestocover50pctvariance", "attributestocover90pctvariance", "attributestocover95pctvariance", "attributestocover99pctvariance", "applicationsize", "fittime", "applicationtime");

	private static final int NUM_TREES = 100;

	private final List<String> parameterFeatures;

	private Map<MetaFeatureContainer, Double> actualInductionTimeCache = new HashMap<>();
	private Map<MetaFeatureContainer, Double> actualInferenceTimeCache = new HashMap<>();

	private final String componentName;

	private final Instances defaultSchema;
	private final Classifier defaultInductionPredictor;
	private final Classifier defaultInferencePredictor;

	private final Instances parameterizedSchema;
	private final Classifier parameterizedInductionPredictor;
	private final Classifier parameterizedInferencePredictor;

	public BaseComponentEvaluationTimePredictor(final String componentName, final KVStoreCollection defaultData, final KVStoreCollection parameterizedData) throws Exception {
		LOGGER.debug("Create component predictor for {}", componentName);
		this.componentName = componentName;

		/* Build models for default configuration */
		LOGGER.debug("Build default configuration model for induction of {}", componentName);
		this.defaultInductionPredictor = this.getModel();

		long start = System.currentTimeMillis();
		Instances defaultInduction = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(defaultData, TARGET_INDUCTION, FEATURES_INDUCTION);
		System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
		this.defaultInductionPredictor.buildClassifier(defaultInduction);

		this.defaultSchema = new Instances(defaultInduction, 0);

		LOGGER.debug("Build default configuration model for inference of {}", componentName);
		this.defaultInferencePredictor = this.getModel();
		start = System.currentTimeMillis();
		Instances defaultInference = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(defaultData, TARGET_INFERENCE, FEATURES_INFERENCE);
		System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
		this.defaultInferencePredictor.buildClassifier(defaultInference);

		if (parameterizedData != null && !parameterizedData.isEmpty()) {
			/* Build models for parameterized configurations */
			this.parameterFeatures = parameterizedData.get(0).keySet().stream().filter(x -> !NON_PARAMETER_COLUMNS.contains(x)).collect(Collectors.toList());

			LOGGER.debug("Build parameterized configuration model for induction of {}", componentName);
			List<String> inductionFeatures = Arrays.stream(FEATURES_INDUCTION).collect(Collectors.toList());
			inductionFeatures.addAll(this.parameterFeatures);

			this.parameterizedInductionPredictor = this.getModel();
			Instances parameterizedInductionData = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(parameterizedData, TARGET_INDUCTION, inductionFeatures);
			this.parameterizedInductionPredictor.buildClassifier(parameterizedInductionData);

			this.parameterizedSchema = new Instances(parameterizedInductionData, 0);

			LOGGER.debug("Build parameterized configuration model for inference of {}", componentName);
			List<String> inferenceFeatures = Arrays.stream(FEATURES_INDUCTION).collect(Collectors.toList());
			inferenceFeatures.addAll(this.parameterFeatures);
			Instances parameterizedInferenceData = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(parameterizedData, TARGET_INDUCTION, inferenceFeatures);
			this.parameterizedInferencePredictor = this.getModel();
			this.parameterizedInferencePredictor.buildClassifier(parameterizedInferenceData);
		} else {
			LOGGER.warn("No data given for parameterized configuraiton model for {}", componentName);
			this.parameterFeatures = null;
			this.parameterizedInductionPredictor = null;
			this.parameterizedInferencePredictor = null;
			this.parameterizedSchema = null;
		}
	}

	private Classifier getModel() {
		RandomForest forest = new RandomForest();
		forest.setNumIterations(NUM_TREES);
		return forest;
	}

	@Override
	public String getComponentName() {
		return this.componentName;
	}

	@Override
	public double predictInductionTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain) throws Exception {
		Classifier model;
		Instance i;
		if (ComponentUtil.isDefaultConfiguration(ci)) {
			if (this.actualInductionTimeCache.containsKey(metaFeaturesTrain)) {
				ci.appendAnnotation(IEvaluationSafeGuard.ANNOTATION_SOURCE, "-IndBLCache");
				double inductionTime = this.actualInductionTimeCache.get(metaFeaturesTrain);
				LOGGER.debug("Return induction time for base component {} from cache: {}.", this.componentName, inductionTime);
				return inductionTime;
			}
			model = this.defaultInductionPredictor;
			ci.appendAnnotation(IEvaluationSafeGuard.ANNOTATION_SOURCE, "-IndBLDef");
			i = this.toDefaultInstance(metaFeaturesTrain);
		} else {
			model = this.parameterizedInductionPredictor;
			ci.appendAnnotation(IEvaluationSafeGuard.ANNOTATION_SOURCE, "-IndBLParam");
			i = this.toParameterizedInstance(metaFeaturesTrain, ci);
		}
		return model.classifyInstance(i);
	}

	@Override
	public double predictInferenceTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain) throws Exception {
		Classifier model;
		Instance i;
		if (ComponentUtil.isDefaultConfiguration(ci)) {
			if (this.actualInferenceTimeCache.containsKey(metaFeaturesTrain)) {
				ci.appendAnnotation(IEvaluationSafeGuard.ANNOTATION_SOURCE, "-InfBLCache");
				double inferenceTime = this.actualInferenceTimeCache.get(metaFeaturesTrain);
				LOGGER.debug("Return inference time for base component {} from cache. {}.", this.componentName, inferenceTime);
				return inferenceTime;
			}
			model = this.defaultInferencePredictor;
			ci.appendAnnotation(IEvaluationSafeGuard.ANNOTATION_SOURCE, "-InfBLDef");
			i = this.toDefaultInstance(metaFeaturesTrain);
		} else {
			model = this.parameterizedInferencePredictor;
			ci.appendAnnotation(IEvaluationSafeGuard.ANNOTATION_SOURCE, "-InfBLParam");
			i = this.toParameterizedInstance(metaFeaturesTrain, ci);
		}
		return model.classifyInstance(i);
	}

	public Instance toDefaultInstance(final MetaFeatureContainer metaFeatureContainer) {
		double[] metaFeatures = metaFeatureContainer.toFeatureVector();
		Instance instance = new DenseInstance(this.defaultSchema.numAttributes());
		for (int i = 0; i < metaFeatures.length; i++) {
			instance.setValue(i, metaFeatures[i]);
		}
		instance.setDataset(this.defaultSchema);
		return instance;
	}

	public Instance toParameterizedInstance(final MetaFeatureContainer metaFeatureContainer, final ComponentInstance ci) {
		double[] metaFeatures = metaFeatureContainer.toFeatureVector();
		Instance instance = new DenseInstance(this.parameterizedSchema.numAttributes());
		instance.setDataset(this.parameterizedSchema);
		int currentI = 0;
		for (currentI = 0; currentI < metaFeatures.length; currentI++) {
			instance.setValue(currentI, metaFeatures[currentI]);
		}

		for (int i = 0; i < this.parameterFeatures.size(); i++) {
			if (this.parameterizedSchema.attribute(currentI).isNumeric()) {
				if (Arrays.asList("true", "false").contains(ci.getParameterValue(this.parameterFeatures.get(i)))) {
					instance.setValue(currentI, ci.getParameterValue(this.parameterFeatures.get(i)).equals("true") ? 1.0 : 0.0);
				} else {
					instance.setValue(currentI, Double.parseDouble(ci.getParameterValue(this.parameterFeatures.get(i))));
				}
			} else {
				instance.setValue(currentI, ci.getParameterValue(this.parameterFeatures.get(i)));
			}
			currentI++;
		}

		return instance;
	}

	@Override
	public void setActualDefaultConfigurationInductionTime(final MetaFeatureContainer metaFeaturesTrain, final double actualInductionTime) {
		this.actualInductionTimeCache.put(metaFeaturesTrain, actualInductionTime);
	}

	@Override
	public void setActualDefaultConfigurationInferenceTime(final MetaFeatureContainer metaFeaturesTrain, final MetaFeatureContainer metaFeaturesTest, final double actualInferenceTime) {
		this.actualInferenceTimeCache.put(metaFeaturesTrain, actualInferenceTime / metaFeaturesTest.getFeature(EMetaFeature.NUM_INSTANCES) * IBaseComponentEvaluationTimePredictor.SCALE_FOR_NUM_PREDICTIONS);
	}

	@Override
	public String toString() {
		Map<String, Object> containedModels = new HashMap<>();
		containedModels.put("defaultInduction", this.defaultInductionPredictor);
		containedModels.put("defaultInference", this.defaultInferencePredictor);
		containedModels.put("paramInduction", this.parameterizedInductionPredictor);
		containedModels.put("paramInference", this.parameterizedInferencePredictor);
		return DataBasedComponentPredictorUtil.safeGuardComponentToString(this.componentName, containedModels);
	}

}
