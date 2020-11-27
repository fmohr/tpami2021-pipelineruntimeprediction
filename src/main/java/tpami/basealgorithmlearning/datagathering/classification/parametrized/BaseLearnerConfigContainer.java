package tpami.basealgorithmlearning.datagathering.classification.parametrized;

import tpami.basealgorithmlearning.AConfigContainer;

public class BaseLearnerConfigContainer extends AConfigContainer {

	public BaseLearnerConfigContainer(final String databaseConfigFile, final String classifierClassName) throws ClassNotFoundException {
		super("conf/experiments/parametrized/baselearner-parametrized-" + classifierClassName.toLowerCase().substring(classifierClassName.lastIndexOf(".") + 1) + ".conf", databaseConfigFile,
				"evaluations_classifiers_" + Class.forName(classifierClassName).getSimpleName().toLowerCase() + "_configured");
	}
}
