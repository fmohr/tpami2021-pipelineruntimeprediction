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

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.components.model.ComponentInstance;
import tpami.basealgorithmlearning.regression.DatasetVarianceFeatureGenerator;
import tpami.safeguard.api.EMetaFeature;
import tpami.safeguard.api.IMetaFeatureTransformationPredictor;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PreprocessingEffectPredictor implements IMetaFeatureTransformationPredictor {

	private static final Logger LOGGER = LoggerFactory.getLogger(PreprocessingEffectPredictor.class);

	private static final Collection<String> NO_PARAMETER_ATTRIBUTES = Arrays.asList("openmlid", "algorithm", "numattributes_before", "numattributes_after", "minAttN");
	private static final String[] FEATURES = { "numattributes_before" };
	private static final String TARGET = "numattributes_after";

	private String componentName;
	private DatasetVarianceFeatureGenerator featureGen;

	/** Predictor for resulting number of attribute after applying this preprocessor with configured hyper-parameters. */
	private List<String> parameterAttributes = null;
	private Instances parameterizedSchema = null;
	private Classifier parameterizedConfigNumAttributes = null;

	public PreprocessingEffectPredictor(final String componentName, final KVStoreCollection transformData) throws Exception {
		this.componentName = componentName;

		this.featureGen = new DatasetVarianceFeatureGenerator();
		this.featureGen.setSuffix("_before");

		// Build model considering parameters.
		if (transformData != null && !transformData.isEmpty()) {
			// Collect columns for parameter features of preprocessor
			this.parameterAttributes = new ArrayList<>(SetUtil.difference(transformData.get(0).keySet(), NO_PARAMETER_ATTRIBUTES));
			List<String> parameterizedFeatures = Arrays.stream(FEATURES).collect(Collectors.toList());
			parameterizedFeatures.addAll(this.parameterAttributes);

			// build dataset & store schema
			long start = System.currentTimeMillis();
			Instances parameterizedDataset = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(transformData, TARGET, parameterizedFeatures);
			System.out.println("Dataset transform needed " + ((double) (System.currentTimeMillis() - start) / 1000) + "s");
			this.parameterizedSchema = new Instances(parameterizedDataset, 0);

			// build model
			this.parameterizedConfigNumAttributes = this.getModel();
			this.parameterizedConfigNumAttributes.buildClassifier(parameterizedDataset);
		} else {
			throw new IllegalArgumentException("No data given!");
		}

	}

	private Classifier getModel() {
		SimpleLinearRegression slr = new SimpleLinearRegression();
		return slr;
	}

	@Override
	public MetaFeatureContainer predictTransformedMetaFeatures(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesBefore) throws Exception {
		double attributesAfter;
		if (this.parameterizedConfigNumAttributes != null) {
			attributesAfter = this.parameterizedConfigNumAttributes.classifyInstance(this.toParameterizedConfigurationInstance(ci, metaFeaturesBefore));
		} else {
			LOGGER.warn("Model for parameterized preprocessor effect not available. Thus use identity function.");
			attributesAfter = metaFeaturesBefore.getFeature(EMetaFeature.NUM_ATTRIBUTES);
		}
		return new MetaFeatureContainer(metaFeaturesBefore.getFeature(EMetaFeature.NUM_INSTANCES), attributesAfter);
	}

	@Override
	public String getComponentName() {
		return this.componentName;
	}

	private Instance toParameterizedConfigurationInstance(final ComponentInstance ci, final MetaFeatureContainer metaFeatureContainer) throws Exception {
		Map<String, Object> varianceFeatures = this.featureGen.getFeatureRepresentation(metaFeatureContainer.getDataset());
		Instance instance = new DenseInstance(this.parameterizedSchema.numAttributes());
		instance.setDataset(this.parameterizedSchema);
		int i = 0;
		instance.setValue(i++, metaFeatureContainer.getFeature(EMetaFeature.NUM_ATTRIBUTES));

		Map<String, Object> parameters = new HashMap<>();
		Queue<ComponentInstance> queue = new LinkedList<>();
		queue.add(ci);
		ComponentInstance element;
		while ((element = queue.poll()) != null) {
			for (Entry<String, String> parameterValue : element.getParameterValues().entrySet()) {
				boolean succeeded = false;

				if (Arrays.asList("true", "false").contains(parameterValue.getValue())) {
					parameters.put(parameterValue.getKey(), parameterValue.getValue().equals("true") ? 1.0 : 0.0);
					continue;
				}

				try {
					parameters.put(parameterValue.getKey(), Double.parseDouble(parameterValue.getValue()));
					succeeded = true;
				} catch (NumberFormatException e) {
					// could not parse to double
				}

				if (!succeeded) {
					try {
						parameters.put(parameterValue.getKey(), (double) Integer.parseInt(parameterValue.getValue()));
						succeeded = true;
					} catch (NumberFormatException e) {
						// could not parse to double
						System.out.println("could not cast parameter value of " + parameterValue.getKey() + " to int");
					}
				}

				if (!succeeded) {
					parameters.put(parameterValue.getKey(), parameterValue.getValue());
					succeeded = true;
				}

			}
			element.getSatisfactionOfRequiredInterfaces().values().forEach(queue::add);
		}

		for (String parameter : this.parameterAttributes) {
			Object value = parameters.get(parameter);
			if (value == null || value.equals("null")) {
				LOGGER.warn("No parameter value for parameter {} for component {}.", parameter, this.componentName);
				i++;
				continue;
			} else if (value instanceof String) {
				instance.setValue(i++, (String) value);
			} else {
				instance.setValue(i++, (Double) value);
			}
		}
		return instance;
	}

	@Override
	public String toString() {
		Map<String, Object> containedModels = new HashMap<>();
		containedModels.put("paramNumAttributes", this.parameterizedConfigNumAttributes);
		return DataBasedComponentPredictorUtil.safeGuardComponentToString(this.componentName, containedModels);
	}
}
