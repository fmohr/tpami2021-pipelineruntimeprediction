package tpami.safeguard;

import ai.libs.hasco.model.ComponentInstance;

public interface IComponentPredictor {

	public String getComponentName();

	public double predictInductionTime(final ComponentInstance ci, final double[] metaFeaturesTrain) throws Exception;

	public double predictInferenceTime(final ComponentInstance ci, final double[] metaFeatureTest) throws Exception;

	default double predictEvaluationTime(final ComponentInstance ci, final double[] metaFeaturesTrain, final double[] metaFeaturesTest) throws Exception {
		return this.predictInductionTime(ci, metaFeaturesTrain) + this.predictInferenceTime(ci, metaFeaturesTest);
	}

	public double getActualDefaultConfigurationInductionTime();

	public void setActualDefaultConfigurationInductionTime(double actualInductionTime);

	public double getActualDefaultConfigurationInferenceTime();

	public void setActualDefaultConfigurationInferenceTime(double actualInferenceTime);

	default void setActualDefaultConfigurationTimes(final Double actualInductionTime, final Double actualInferenceTime) {
		if (actualInductionTime != null) {
			this.setActualDefaultConfigurationInductionTime(actualInductionTime);
		}
		if (actualInferenceTime != null) {
			this.setActualDefaultConfigurationInferenceTime(actualInferenceTime);
		}
	}

}
