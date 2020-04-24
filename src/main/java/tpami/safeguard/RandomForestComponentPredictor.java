package tpami.safeguard;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class RandomForestComponentPredictor implements IComponentPredictor {

	private static final Logger LOGGER = LoggerFactory.getLogger(RandomForestComponentPredictor.class);

	private static final String TARGET_INDUCTION = "fittime";
	private static final String TARGET_INFERENCE = "applicationtime";
	private static final String[] FEATURES_INDUCTION = { "fitsize", "fitattributes" };
	private static final String[] FEATURES_INFERENCE = { "applicationsize", "fitattributes" };
	private static final List<String> NON_PARAMETER_COLUMNS = Arrays.asList("openmlid", "totalsize", "fitsize", "applicationsize", "fitattributes", "seed", "algorithm", "fittime", "applicationtime");

	private static final int NUM_TREES = 100;

	private final List<String> parameterFeatures;

	private Double actualInductionTime = null;
	private Double actualInferenceTime = null;

	private final String componentName;

	private final Instances defaultSchema;
	private final Classifier defaultInductionPredictor;
	private final Classifier defaultInferencePredictor;

	private final Instances parameterizedSchema;
	private final Classifier parameterizedInductionPredictor;
	private final Classifier parameterizedInferencePredictor;

	public RandomForestComponentPredictor(final String componentName, final KVStoreCollection defaultData, final KVStoreCollection parameterizedData) throws Exception {
		LOGGER.debug("Create component predictor for {}", componentName);
		this.componentName = componentName;

		/* Build models for default configuration */
		LOGGER.debug("Build default configuration model for induction of {}", componentName);
		this.defaultInductionPredictor = this.getModel();
		Instances defaultInduction = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(defaultData, TARGET_INDUCTION, FEATURES_INDUCTION);
		this.defaultInductionPredictor.buildClassifier(defaultInduction);

		this.defaultSchema = new Instances(defaultInduction, 0);

		LOGGER.debug("Build default configuration model for inference of {}", componentName);
		this.defaultInferencePredictor = this.getModel();
		Instances defaultInference = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(defaultData, TARGET_INFERENCE, FEATURES_INFERENCE);
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
	public double predictInductionTime(final ComponentInstance ci, final double[] metaFeaturesTrain) throws Exception {
		Classifier model;
		Instance i;
		if (ComponentUtil.isDefaultConfiguration(ci)) {
			model = this.defaultInductionPredictor;
			i = this.toDefaultInstance(metaFeaturesTrain);
		} else {
			model = this.parameterizedInductionPredictor;
			i = this.toParameterizedInstance(metaFeaturesTrain, ci);
		}
		return model.classifyInstance(i);
	}

	@Override
	public double predictInferenceTime(final ComponentInstance ci, final double[] metaFeaturesTest) throws Exception {
		Classifier model;
		Instance i;
		if (ComponentUtil.isDefaultConfiguration(ci)) {
			model = this.defaultInferencePredictor;
			i = this.toDefaultInstance(metaFeaturesTest);
		} else {
			model = this.parameterizedInferencePredictor;
			i = this.toParameterizedInstance(metaFeaturesTest, ci);
		}
		return model.classifyInstance(i);
	}

	public Instance toDefaultInstance(final double[] metaFeatures) {
		Instance instance = new DenseInstance(this.defaultSchema.numAttributes());
		for (int i = 0; i < metaFeatures.length; i++) {
			instance.setValue(i, metaFeatures[i]);
		}
		instance.setDataset(this.defaultSchema);
		return instance;
	}

	public Instance toParameterizedInstance(final double[] metaFeatures, final ComponentInstance ci) {
		Instance instance = new DenseInstance(this.parameterizedSchema.numAttributes());
		int currentI = 0;
		for (currentI = 0; currentI < metaFeatures.length; currentI++) {
			instance.setValue(currentI, metaFeatures[currentI]);
		}

		for (int i = 0; i < this.parameterFeatures.size(); i++) {
			if (this.parameterizedSchema.attribute(currentI).isNumeric()) {
				instance.setValue(currentI, Double.parseDouble(ci.getParameterValue(this.parameterFeatures.get(i))));
			} else {
				instance.setValue(currentI, ci.getParameterValue(this.parameterFeatures.get(i)));
			}
			currentI++;
		}

		instance.setDataset(this.parameterizedSchema);
		return instance;
	}

	@Override
	public void setActualDefaultConfigurationInductionTime(final double actualInductionTime) {
		this.actualInductionTime = actualInductionTime;
	}

	@Override
	public double getActualDefaultConfigurationInductionTime() {
		return this.actualInductionTime;
	}

	@Override
	public void setActualDefaultConfigurationInferenceTime(final double actualInferenceTime) {
		this.actualInferenceTime = actualInferenceTime;
	}

	@Override
	public double getActualDefaultConfigurationInferenceTime() {
		return this.actualInferenceTime;
	}

}
