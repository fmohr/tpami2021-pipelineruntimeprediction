package tpami.safeguard.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.components.model.CategoricalParameterDomain;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.NumericParameterDomain;
import ai.libs.jaicore.components.model.Parameter;
import tpami.safeguard.api.IBaseComponentEvaluationTimePredictor;
import tpami.safeguard.api.IMetaLearnerEvaluationTimePredictor;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import tpami.safeguard.util.MLComponentInstanceWrapper;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MetaLearnerEvaluationTimePredictor implements IMetaLearnerEvaluationTimePredictor {

	private static final List<String> NON_PARAMETER_ATTRIBUTES = Arrays.asList("openmlid", "algorithm", "algorithmoptions", "numinstances", "numattributes", "numattributesafterbinarization", "trainpoints", "numinstances_sub",
			"numattributes_sub", "builds", "predictioncalls_training", "predictioncalls_prediction");

	private static final String[] FEATURES_A = { "trainpoints", "numattributesafterbinarization" };
	private static final String[] FEATURES_B = { "trainpoints", "numattributes" };

	private static final String TARGET_SUB_NUMINSTANCES = "numinstances_sub";
	private static final String TARGET_SUB_NUMATTRIBUTES = "numattributes_sub";

	private static final String TARGET_BUILDS = "builds";
	private static final String TARGET_BL_CALLS_INDUCTION = "predictioncalls_training";
	private static final String TARGET_BL_CALLS_INFERENCE = "predictioncalls_prediction";

	private final String componentName;
	private List<String> parameters;
	private List<String> features;

	private final Instances schema;
	private final Classifier numBaseLearnerBuilds;
	private final Classifier numInstances;
	private final Classifier numAttributes;
	private final Classifier inductionNumBaseLearnerInferences;
	private final Classifier inferenceNumBaseLearnerInferences;

	public MetaLearnerEvaluationTimePredictor(final String componentName, final KVStoreCollection metaLearnerEffectData) throws Exception {
		this.componentName = componentName;
		this.parameters = new ArrayList<>(SetUtil.difference(metaLearnerEffectData.get(0).keySet(), NON_PARAMETER_ATTRIBUTES));
		this.features = new ArrayList<>(Arrays.stream(FEATURES_A).collect(Collectors.toList()));
		this.features.addAll(this.parameters);

		// generic schema for accessing the meta classifier models
		ArrayList<Attribute> attributeList = new ArrayList<>();
		this.features.stream().map(x -> new Attribute(x)).forEach(attributeList::add);
		attributeList.add(new Attribute("target"));

		this.schema = new Instances("general-schema", attributeList, 0);
		this.schema.setClassIndex(this.schema.numAttributes() - 1);

		// meta classifier behavior models
		this.numBaseLearnerBuilds = this.getModel();
		long start = System.currentTimeMillis();
		this.numBaseLearnerBuilds.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_BUILDS, this.features));
		System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
		this.inductionNumBaseLearnerInferences = this.getModel();
		start = System.currentTimeMillis();
		this.inductionNumBaseLearnerInferences.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_BL_CALLS_INDUCTION, this.features));
		System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
		this.inferenceNumBaseLearnerInferences = this.getModel();
		start = System.currentTimeMillis();
		this.inferenceNumBaseLearnerInferences.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_BL_CALLS_INFERENCE, this.features));
		System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");

		// meta feature transformation models
		this.numInstances = this.getModel();
		start = System.currentTimeMillis();
		this.numInstances.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_SUB_NUMINSTANCES, this.features));
		System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
		this.numAttributes = this.getModel();
		start = System.currentTimeMillis();
		this.numAttributes.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_SUB_NUMATTRIBUTES, this.features));
		System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");

	}

	private Classifier getModel() {
		RandomForest rf = new RandomForest();
		rf.setNumIterations(100);
		return rf;
	}

	private Instance toInstance(final ComponentInstance ci, final MetaFeatureContainer metaFeatureContainer) {
		double[] metaFeatures = metaFeatureContainer.toFeatureVector();
		Instance instance = new DenseInstance(this.schema.numAttributes());
		for (int i = 0; i < metaFeatures.length; i++) {
			instance.setValue(i, metaFeatures[i]);
		}

		Map<String, Object> paramValueMap = new HashMap<>();
		for (String paramName : this.parameters) {
			Parameter param = ci.getComponent().getParameterWithName(paramName);
			if (param.getDefaultDomain() instanceof CategoricalParameterDomain) {
				paramValueMap.put(paramName, ci.getParameterValue(paramName));
			} else if (param.getDefaultDomain() instanceof NumericParameterDomain) {
				paramValueMap.put(paramName, Double.parseDouble(ci.getParameterValue(paramName)));
			}
		}

		instance.setDataset(this.schema);
		return instance;
	}

	@Override
	public double predictInductionTime(final MLComponentInstanceWrapper ciw, final IBaseComponentEvaluationTimePredictor baseLearnerEvaluationTimePredictor, final MetaFeatureContainer metaFeaturesTrain) throws Exception {
		Objects.requireNonNull(baseLearnerEvaluationTimePredictor, "Baselearner evaluation time predictor must not be null.");
		Objects.requireNonNull(ciw.getBaseLearner(), "Base learner must not be null.");

		Instance instance = this.toInstance(ciw, metaFeaturesTrain);

		double k = this.numBaseLearnerBuilds.classifyInstance(instance);
		double predictionCallsOfBaseLearnerDuringInduction = this.inductionNumBaseLearnerInferences.classifyInstance(instance);

		double numSubInstances = this.numInstances.classifyInstance(instance);
		double numSubAttributes = this.numAttributes.classifyInstance(instance);
		MetaFeatureContainer subMetaFeatures = new MetaFeatureContainer(numSubInstances, numSubAttributes);
		double baseLearnerInductionTime = baseLearnerEvaluationTimePredictor.predictInductionTime(ciw.getBaseLearner(), subMetaFeatures);
		double baseLearnerInferenceTime = baseLearnerEvaluationTimePredictor.predictInferenceTime(ciw.getBaseLearner(), subMetaFeatures) / IBaseComponentEvaluationTimePredictor.SCALE_FOR_NUM_PREDICTIONS;

		return k * (baseLearnerInductionTime + predictionCallsOfBaseLearnerDuringInduction * baseLearnerInferenceTime);
	}

	@Override
	public double predictInferenceTime(final MLComponentInstanceWrapper ciw, final IBaseComponentEvaluationTimePredictor baseLearnerEvaluationTimePredictor, final MetaFeatureContainer metaFeaturesTrain,
			final MetaFeatureContainer metaFeaturesTest) throws Exception {
		Instance instance = this.toInstance(ciw, metaFeaturesTrain);

		double k = this.numBaseLearnerBuilds.classifyInstance(instance);
		double predictionCallsOfBaseLearnerDuringInference = this.inferenceNumBaseLearnerInferences.classifyInstance(instance);

		double numSubInstances = this.numInstances.classifyInstance(instance);
		double numSubAttributes = this.numAttributes.classifyInstance(instance);
		MetaFeatureContainer subMetaFeatures = new MetaFeatureContainer(numSubInstances, numSubAttributes);
		double baseLearnerInferenceTime = baseLearnerEvaluationTimePredictor.predictInferenceTime(ciw.getBaseLearner(), subMetaFeatures, predictionCallsOfBaseLearnerDuringInference);

		return k * baseLearnerInferenceTime;
	}

	@Override
	public String toString() {
		Map<String, Object> containedModels = new HashMap<>();
		containedModels.put("builds", this.numBaseLearnerBuilds);
		containedModels.put("subAttributes", this.numAttributes);
		containedModels.put("subInstances", this.numInstances);
		containedModels.put("inferencesDuringTraining", this.inductionNumBaseLearnerInferences);
		containedModels.put("inferencesDuringApplication", this.inferenceNumBaseLearnerInferences);
		return DataBasedComponentPredictorUtil.safeGuardComponentToString(this.componentName, containedModels);
	}

}
