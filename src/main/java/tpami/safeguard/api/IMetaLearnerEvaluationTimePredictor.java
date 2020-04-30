package tpami.safeguard.api;

import tpami.safeguard.impl.MetaFeatureContainer;
import tpami.safeguard.util.MLComponentInstanceWrapper;

/**
 * This model can be used to predict the runtimes of a meta learner which incorporates a base learner.
 * The model for the respective base learner needs to be provided to the meta learner.
 *
 * @author mwever
 */
public interface IMetaLearnerEvaluationTimePredictor {

	/**
	 * Predicts the runtime that is needed for inducing the meta learner component instance for the training dataset.
	 *
	 * @param ciw The component instance wrapper desribing the meta learner.
	 * @param iComponentPredictor The component evaluation time predictor of the base learner.
	 * @param metaFeaturesTrain The meta features describing the shape of the training dataset.
	 * @return The runtime of the meta learner (including running its base learner).
	 * @throws Exception
	 */
	public double predictInductionTime(MLComponentInstanceWrapper ciw, IBaseComponentEvaluationTimePredictor iComponentPredictor, MetaFeatureContainer metaFeaturesTrain) throws Exception;

	/**
	 * Predicts the runtime that is needed for applying the meta learner component instance to a test dataset.
	 *
	 * @param ciw The component instance wrapper desribing the meta learner.
	 * @param iComponentPredictor The component evaluation time predictor of the base learner.
	 * @param metaFeaturesTrain The meta features describing the shape of the training dataset.
	 * @param metaFeaturesTest The meta features describing the shape of the test dataset.
	 * @return The runtime of the meta learner (including running its base learner).
	 * @throws Exception
	 */
	public double predictInferenceTime(MLComponentInstanceWrapper ciw, IBaseComponentEvaluationTimePredictor iComponentPredictor, MetaFeatureContainer metaFeaturesTrain, MetaFeatureContainer metaFeaturesTest) throws Exception;

	/**
	 * Predicts the runtime that is needed for evaluating, i.e. induction and inference, the meta learner component instance for a training dataset and a test dataset as described via the meta features.
	 *
	 * @param ciw The component instance wrapper desribing the meta learner.
	 * @param iComponentPredictor The component evaluation time predictor of the base learner.
	 * @param metaFeaturesTrain The meta features describing the shape of the training dataset.
	 * @param metaFeaturesTest The meta features describing the shape of the test dataset.
	 * @return The runtime of the meta learner (including running its base learner).
	 * @throws Exception
	 */
	default double predictEvaluationTime(final MLComponentInstanceWrapper ciw, final IBaseComponentEvaluationTimePredictor iComponentPredictor, final MetaFeatureContainer metaFeaturesTrain, final MetaFeatureContainer metaFeaturesTest)
			throws Exception {
		return this.predictInductionTime(ciw, iComponentPredictor, metaFeaturesTrain) + this.predictInferenceTime(ciw, iComponentPredictor, metaFeaturesTrain, metaFeaturesTest);
	}

}
