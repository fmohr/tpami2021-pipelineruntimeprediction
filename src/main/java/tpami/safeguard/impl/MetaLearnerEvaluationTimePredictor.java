package tpami.safeguard.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.libs.hasco.model.CategoricalParameterDomain;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.NumericParameterDomain;
import ai.libs.hasco.model.Parameter;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
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

	private List<String> parameters;

	private final Instances schema;
	private final Classifier numBaseLearnerBuilds;
	private final Classifier numInstances;
	private final Classifier numAttributes;
	private final Classifier inductionNumBaseLearnerInferences;
	private final Classifier inferenceNumBaseLearnerInferences;

	public MetaLearnerEvaluationTimePredictor(final KVStoreCollection metaLearnerEffectData) throws Exception {
		// generic schema for accessing the meta classifier models
		ArrayList<Attribute> attributeList = new ArrayList<>();
		attributeList.add(new Attribute("numinstances"));
		attributeList.add(new Attribute("numattributes"));
		attributeList.add(new Attribute("target"));
		this.schema = new Instances("general-schema", attributeList, 0);

		// meta classifier behavior models
		this.numBaseLearnerBuilds = this.getModel();
		this.numBaseLearnerBuilds.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_BUILDS, FEATURES_A));
		this.inductionNumBaseLearnerInferences = this.getModel();
		this.inductionNumBaseLearnerInferences.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_BL_CALLS_INDUCTION, FEATURES_A));
		this.inferenceNumBaseLearnerInferences = this.getModel();
		this.inferenceNumBaseLearnerInferences.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_BL_CALLS_INFERENCE, FEATURES_A));

		// meta feature transformation models
		this.numInstances = this.getModel();
		this.numInstances.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_SUB_NUMINSTANCES, FEATURES_A));
		this.numAttributes = this.getModel();
		this.numAttributes.buildClassifier(DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(metaLearnerEffectData, TARGET_SUB_NUMATTRIBUTES, FEATURES_A));

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

		return instance;
	}

	@Override
	public double predictInductionTime(final MLComponentInstanceWrapper ciw, final IBaseComponentEvaluationTimePredictor baseLearnerEvaluationTimePredictor, final MetaFeatureContainer metaFeaturesTrain) throws Exception {
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

}
