package tpami.basealgorithmlearning.datagathering.classification.defaultparams;

import tpami.basealgorithmlearning.AConfigContainer;

public class DefaultBaseLearnerConfigContainer extends AConfigContainer {

	public DefaultBaseLearnerConfigContainer(final String databaseConfigFile, final String classifierClassName) throws ClassNotFoundException {
		super("conf/experiments/defaultparams/baselearner.conf", databaseConfigFile, "evaluations_classifiers_" + Class.forName(classifierClassName).getSimpleName().toLowerCase());
	}

	public IDefaultBaseLearnerExperimentConfig getConfig() {
		return (IDefaultBaseLearnerExperimentConfig)this.config;
	}
}
