package tpami.safeguard.util;

import ai.libs.jaicore.components.model.ComponentInstance;

public class MLComponentInstanceWrapper extends ComponentInstance {

	public MLComponentInstanceWrapper(final ComponentInstance ci) {
		super(ci);
	}

	public boolean isPreprocessor() {
		return this.getComponent().getProvidedInterfaces().contains("AbstractPreprocessor");
	}

	public boolean isMetaLearner() {
		return this.getComponent().getName().contains("meta");
	}

	public boolean isBaseLearner() {
		return !(this.isPipeline() || this.isMetaLearner());
	}

	public boolean isPipeline() {
		return this.getComponent().getName().equals("pipeline");
	}

	public MLComponentInstanceWrapper getPreprocessor() {
		if (this.isPipeline()) {
			return new MLComponentInstanceWrapper(this.getSatisfactionOfRequiredInterfaces().get("preprocessor"));
		}
		return null;
	}

	public MLComponentInstanceWrapper getClassifier() {
		if (this.isPipeline()) {
			return new MLComponentInstanceWrapper(this.getSatisfactionOfRequiredInterfaces().get("classifier"));
		} else {
			return this;
		}
	}

	public MLComponentInstanceWrapper getBaseLearner() {
		if (this.isPipeline() && !this.getClassifier().isMetaLearner()) {
			return this.getClassifier();
		} else if (this.isPipeline() && this.getClassifier().isMetaLearner()) {
			return this.getClassifier().getBaseLearner();
		} else if (this.getClassifier().isMetaLearner()) {
			return new MLComponentInstanceWrapper(this.getSatisfactionOfRequiredInterfaces().get("W"));
		} else {
			return this;
		}
	}

}
