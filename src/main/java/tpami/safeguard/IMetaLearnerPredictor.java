package tpami.safeguard;

import ai.libs.hasco.model.ComponentInstance;

public interface IMetaLearnerPredictor {

	public double predictInductionTime(ComponentInstance ciw, IComponentPredictor iComponentPredictor, double[] metaFeaturesTrain);

	public double predictInferenceTime(MLComponentInstanceWrapper ciw, IComponentPredictor iComponentPredictor, double[] metaFeaturesTest);

}
