package tpami.safeguard.util;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstance;

public class MLComponentInstanceWrapper extends ComponentInstance {

	public MLComponentInstanceWrapper(final IComponentInstance ci) {
		super((ComponentInstance)ci);
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
			return new MLComponentInstanceWrapper(this.getSatisfactionOfRequiredInterface("preprocessor").get(0));
		}
		return null;
	}

	public MLComponentInstanceWrapper getClassifier() {
		if (this.isPipeline()) {
			return new MLComponentInstanceWrapper(this.getSatisfactionOfRequiredInterface("classifier").get(0));
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
			return new MLComponentInstanceWrapper((ComponentInstance)this.getSatisfactionOfRequiredInterfaces().get("W"));
		} else {
			return this;
		}
	}

}
