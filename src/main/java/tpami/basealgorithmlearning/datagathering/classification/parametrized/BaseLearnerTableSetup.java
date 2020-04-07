package tpami.basealgorithmlearning.datagathering.classification.parametrized;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.ml.weka.WekaUtil;

public class BaseLearnerTableSetup {

	public static void main(final String[] args) throws Exception {
		for (String baseLearner: WekaUtil.getBasicLearners()) {

			System.out.println(baseLearner);
			if (!baseLearner.contains("weka.classifiers.functions.VotedPerceptron")) {
				continue;
			}

			/* prepare database for this combination */
			BaseLearnerConfigContainer container = new BaseLearnerConfigContainer("conf/dbcon-local.conf", baseLearner);
			ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
			preparer.setLoggerName("example");
			preparer.synchronizeExperiments();
		}
	}
}
