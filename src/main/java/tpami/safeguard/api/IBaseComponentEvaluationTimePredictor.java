package tpami.safeguard.api;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

import ai.libs.hasco.model.ComponentInstance;
import tpami.safeguard.impl.MetaFeatureContainer;

/**
 * This model can be used for predicting induction and inference runtimes for a basic component, i.e., a basic classifier or a preprocessor.
 *
 * @author mwever
 */
public interface IBaseComponentEvaluationTimePredictor {

	public static final int SCALE_FOR_NUM_PREDICTIONS = 1000;

	public String getComponentName();

	public double predictInductionTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain) throws Exception;

	/**
	 * Predicts the runtime of the provided component instance for a test set of SCALE_FOR_NUM_PREDICTIONS many instances.
	 *
	 * @param ci The component instance for which to predict the runtime of the inference phase for.
	 * @param metaFeaturesTrain The meta features describing the shape of the training data.
	 * @return The predicted runtime for inferring on a test set of SCALE_FOR_NUM_PREDICTIONS many instances.
	 *
	 * @throws Exception
	 */
	public double predictInferenceTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain) throws Exception;

	/**
	 * Predicts the time needed for inference on the number of test instances as described in {@link metaFeaturesTest}.
	 *
	 * @param ci The component instance for which to predict the runtime of the inference phase for.
	 * @param metaFeaturesTrain The meta features describing the shape of the training data (for which the component instance was built).
	 * @param metaFeaturesTest The meta features describing the shape of the test data (for which the runtime shall be predicted).
	 * @return The runtime for the described test data.
	 *
	 * @throws Exception
	 */
	default double predictInferenceTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain, final MetaFeatureContainer metaFeaturesTest) throws Exception {
		return this.predictInferenceTime(ci, metaFeaturesTrain, metaFeaturesTest.getFeature(EMetaFeature.NUM_INSTANCES));
	}

	/**
	 * Predicts the time needed for inference on the number of test instances as described in {@link metaFeaturesTest}.
	 *
	 * @param ci The component instance for which to predict the runtime of the inference phase.
	 * @param metaFeaturesTrain The meta features describing the shape of the training data (for which the component instance was built).
	 * @param numTestInstances The number of test instances to calculate the runtime for.
	 * @return The runtime for inferring the given number of test instances.
	 *
	 * @throws Exception
	 */
	default double predictInferenceTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain, final double numTestInstances) throws Exception {
		return this.predictInferenceTime(ci, metaFeaturesTrain) / SCALE_FOR_NUM_PREDICTIONS * numTestInstances;
	}

	/**
	 * Predicts the runtime needed for both induction and inference for the datasets as described through the meta features provided.
	 *
	 * @param ci The component instance for which to predict the runtime.
	 * @param metaFeaturesTrain The meta features describing the shape of the training data (for which the component instance is to be built).
	 * @param metaFeaturesTest The meta features describing the shape of the test data (for which the the component instance is to be applied to).
	 * @return The total runtime needed for inducing the component instance and using the resulting object to apply it to the test data.
	 *
	 * @throws Exception
	 */
	default double predictEvaluationTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain, final MetaFeatureContainer metaFeaturesTest) throws Exception {
		return this.predictInductionTime(ci, metaFeaturesTrain) + this.predictInferenceTime(ci, metaFeaturesTrain, metaFeaturesTest);
	}

	/**
	 * Predicts the runtime needed for both induction and inference for the datasets as described through the meta features provided.
	 *
	 * @param ci The component instance for which to predict the runtime.
	 * @param metaFeaturesTrain The meta features describing the shape of the training data (for which the component instance is to be built).
	 * @param metaFeaturesTest The meta features describing the shape of the test data (for which the the component instance is to be applied to).
	 * @return The total runtime needed for inducing the component instance and using the resulting object to apply it to the test data.
	 *
	 * @throws Exception
	 */
	default double predictEvaluationTime(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesTrain, final double numTestInstances) throws Exception {
		return this.predictInductionTime(ci, metaFeaturesTrain) + this.predictInferenceTime(ci, metaFeaturesTrain, numTestInstances);
	}

	public void setActualDefaultConfigurationInductionTime(MetaFeatureContainer metaFeaturesTrain, double actualInductionTime);

	public void setActualDefaultConfigurationInferenceTime(MetaFeatureContainer metaFeaturesTrain, MetaFeatureContainer metaFeaturesTest, double actualInferenceTime);

	default void setActualDefaultConfigurationTimes(final MetaFeatureContainer metaFeaturesTrain, final MetaFeatureContainer metaFeaturesTest, final Double actualInductionTime, final Double actualInferenceTime) {
		if (actualInductionTime != null) {
			this.setActualDefaultConfigurationInductionTime(metaFeaturesTrain, actualInductionTime);
		}
		if (actualInferenceTime != null) {
			this.setActualDefaultConfigurationInferenceTime(metaFeaturesTrain, metaFeaturesTest, actualInferenceTime);
		}
	}

	default void setActualDefaultConfigurationTimes(final ILabeledDataset<?> dTrain, final ILabeledDataset<?> dTest, final Double actualInductionTime, final Double actualInferenceTime) {
		this.setActualDefaultConfigurationTimes(new MetaFeatureContainer(dTrain), new MetaFeatureContainer(dTest), actualInductionTime, actualInferenceTime);
	}

	default void setActualDefaultConfigurationInductionTime(final ILabeledDataset<?> dTrain, final double actualInductionTime) {
		this.setActualDefaultConfigurationInductionTime(new MetaFeatureContainer(dTrain), actualInductionTime);
	}

	default void setActualDefaultConfigurationInferenceTime(final ILabeledDataset<?> dTrain, final ILabeledDataset<?> dTest, final double actualInferenceTime) {
		this.setActualDefaultConfigurationInferenceTime(new MetaFeatureContainer(dTrain), new MetaFeatureContainer(dTest), actualInferenceTime);
	}
}
