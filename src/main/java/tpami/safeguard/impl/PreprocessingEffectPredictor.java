package tpami.safeguard.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Queue;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.hasco.model.CategoricalParameterDomain;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.model.Parameter;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.sets.SetUtil;
import tpami.basealgorithmlearning.regression.DatasetVarianceFeatureGenerator;
import tpami.safeguard.api.EMetaFeature;
import tpami.safeguard.api.IMetaFeatureTransformationPredictor;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PreprocessingEffectPredictor implements IMetaFeatureTransformationPredictor {

	private static final Logger LOGGER = LoggerFactory.getLogger(PreprocessingEffectPredictor.class);

	private static final Collection<String> NO_PARAMETER_ATTRIBUTES = Arrays.asList("openmlid", "algorithm", "algorithmoptions", "numinstances", "numattributes", "numattributesafterbinarization", "trainpoints", "numinstances_sub",
			"numattributes_sub", "builds", "predictioncalls_training", "predictioncalls_prediction");
	private static final String[] FEATURES = { "numattributes_before", "attributestocover50pctvariance_before", "attributestocover99pctvariance_before" };
	private static final String TARGET = "numattributes_after";

	private String componentName;
	private DatasetVarianceFeatureGenerator featureGen;

	/** Predictor for resulting number of attributes after applying this preprocessor in default configuration. */
	private Instances defaultSchema;
	private Classifier defaultConfigNumAttributes;

	/** Predictor for resulting number of attribute after applying this preprocessor with configured hyper-parameters. */
	private List<String> parameterAttributes = null;
	private Instances parameterizedSchema = null;
	private Classifier parameterizedConfigNumAttributes = null;

	public PreprocessingEffectPredictor(final String componentName, final KVStoreCollection defaultParams, final KVStoreCollection parameterizedConfigurationCol) throws Exception {
		this.componentName = componentName;

		// Build default parameterization model
		Instances defaultDataset = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(defaultParams, TARGET, FEATURES);
		this.defaultSchema = new Instances(defaultDataset, 0);
		this.defaultConfigNumAttributes = this.getModel();
		this.defaultConfigNumAttributes.buildClassifier(defaultDataset);

		// Build model considering parameters.
		if (parameterizedConfigurationCol != null && !parameterizedConfigurationCol.isEmpty()) {
			this.parameterAttributes = new ArrayList<>(SetUtil.difference(parameterizedConfigurationCol.get(0).keySet(), NO_PARAMETER_ATTRIBUTES));
			List<String> parameterizedFeatures = Arrays.stream(FEATURES).collect(Collectors.toList());
			parameterizedFeatures.addAll(this.parameterAttributes);
			Instances parameterizedDataset = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(parameterizedConfigurationCol, TARGET, parameterizedFeatures);
			this.parameterizedSchema = new Instances(parameterizedDataset, 0);
			this.parameterizedConfigNumAttributes.buildClassifier(parameterizedDataset);
		}

		this.featureGen = new DatasetVarianceFeatureGenerator();
		this.featureGen.setSuffix("_before");
	}

	private Classifier getModel() {
		RandomForest rf = new RandomForest();
		rf.setNumIterations(100);
		return rf;
	}

	@Override
	public MetaFeatureContainer predictTransformedMetaFeatures(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesBefore) throws Exception {
		double attributesAfter;
		if (ComponentUtil.isDefaultConfiguration(ci)) {
			attributesAfter = this.defaultConfigNumAttributes.classifyInstance(this.toDefaultConfigurationInstance(metaFeaturesBefore));
		} else {
			if (this.parameterizedConfigNumAttributes != null) {
				attributesAfter = this.parameterizedConfigNumAttributes.classifyInstance(this.toParameterizedConfigurationInstance(ci, metaFeaturesBefore));
			} else {
				LOGGER.warn("Model for parameterized preprocessor effect not available. Thus use identity function.");
				attributesAfter = metaFeaturesBefore.getFeature(EMetaFeature.NUM_ATTRIBUTES);
			}
		}
		return new MetaFeatureContainer(metaFeaturesBefore.getFeature(EMetaFeature.NUM_INSTANCES), attributesAfter);
	}

	@Override
	public String getComponentName() {
		return this.componentName;
	}

	private Instance toDefaultConfigurationInstance(final MetaFeatureContainer metaFeatureContainer) throws Exception {
		Map<String, Object> varianceFeatures = this.featureGen.getFeatureRepresentation(metaFeatureContainer.getDataset());
		Instance instance = new DenseInstance(this.defaultSchema.numAttributes());
		instance.setValue(0, metaFeatureContainer.getFeature(EMetaFeature.NUM_ATTRIBUTES));
		instance.setValue(1, (Integer) varianceFeatures.get(FEATURES[1]));
		instance.setValue(2, (Integer) varianceFeatures.get(FEATURES[2]));
		instance.setDataset(this.defaultSchema);
		return instance;
	}

	private Instance toParameterizedConfigurationInstance(final ComponentInstance ci, final MetaFeatureContainer metaFeatureContainer) throws Exception {
		Map<String, Object> varianceFeatures = this.featureGen.getFeatureRepresentation(metaFeatureContainer.getDataset());
		Instance instance = new DenseInstance(this.parameterizedSchema.numAttributes());
		int i = 0;
		instance.setValue(i++, metaFeatureContainer.getFeature(EMetaFeature.NUM_ATTRIBUTES));
		instance.setValue(i++, (Double) varianceFeatures.get(FEATURES[1]));
		instance.setValue(i++, (Double) varianceFeatures.get(FEATURES[2]));

		Map<String, Object> parameters = new HashMap<>();
		Queue<ComponentInstance> queue = new LinkedList<>();
		queue.add(ci);
		ComponentInstance element;
		while ((element = queue.poll()) != null) {
			for (Entry<String, String> parameterValue : element.getParameterValues().entrySet()) {
				Parameter param = element.getComponent().getParameterWithName(parameterValue.getKey());
				if (param.getDefaultDomain() instanceof CategoricalParameterDomain) {
					parameters.put(parameterValue.getKey(), parameterValue.getValue());
				} else {
					try {
						parameters.put(parameterValue.getKey(), Double.parseDouble(parameterValue.getValue()));
					} catch (NumberFormatException e) {
						parameters.put(parameterValue.getKey(), (double) Integer.parseInt(parameterValue.getValue()));
					}
				}
			}
			element.getSatisfactionOfRequiredInterfaces().values().forEach(queue::add);
		}

		for (String parameter : this.parameterAttributes) {
			Object value = parameters.get(parameter);
			if (value == null) {
				LOGGER.warn("No parameter value for parameter {} for component {}.", parameter, this.componentName);
				i++;
				continue;
			} else if (value instanceof String) {
				instance.setValue(i++, (String) value);
			} else {
				instance.setValue(i++, (Double) value);
			}
		}
		instance.setDataset(this.parameterizedSchema);
		return instance;
	}

	@Override
	public String toString() {
		Map<String, Object> containedModels = new HashMap<>();
		containedModels.put("defaultNumAttributes", this.defaultConfigNumAttributes);
		containedModels.put("paramNumAttributes", this.parameterizedConfigNumAttributes);
		return DataBasedComponentPredictorUtil.safeGuardComponentToString(this.componentName, containedModels);
	}
}
