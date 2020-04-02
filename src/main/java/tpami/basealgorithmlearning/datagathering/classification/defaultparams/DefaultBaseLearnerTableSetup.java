package tpami.basealgorithmlearning.datagathering.classification.defaultparams;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.ml.weka.WekaUtil;

public class DefaultBaseLearnerTableSetup {

	public static void main(final String[] args) throws Exception {
		for (String baseLearner: WekaUtil.getBasicLearners()) {

			System.out.println(baseLearner);
			if (!baseLearner.contains("weka.classifiers.functions.Logistic")) {
				continue;
			}

			/* prepare database for this combination */
			DefaultBaseLearnerConfigContainer container = new DefaultBaseLearnerConfigContainer("conf/dbcon-local.conf", baseLearner);
			ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
			preparer.setLoggerName("example");
			preparer.synchronizeExperiments();
		}
	}
}
