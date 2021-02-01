package tpami.basealgorithmlearning.datagathering.classification.defaultparams;

import tpami.basealgorithmlearning.AConfigContainer;
import tpami.basealgorithmlearning.datagathering.ILearnerExperimentConfig;

public class DefaultBaseLearnerConfigContainer extends AConfigContainer {

	public DefaultBaseLearnerConfigContainer(final String databaseConfigFile, final String experimentSetupFile, final String classifierClassName) throws ClassNotFoundException {
		super(experimentSetupFile, databaseConfigFile, "evaluations_classifiers_" + Class.forName(classifierClassName).getSimpleName().toLowerCase());
	}

	public ILearnerExperimentConfig getConfig() {
		return (ILearnerExperimentConfig)this.config;
	}
}
