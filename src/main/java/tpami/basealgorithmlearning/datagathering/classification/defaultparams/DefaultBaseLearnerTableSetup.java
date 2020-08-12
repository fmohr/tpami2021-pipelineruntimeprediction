package tpami.basealgorithmlearning.datagathering.classification.defaultparams;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.ml.weka.WekaUtil;

public class DefaultBaseLearnerTableSetup {

	public static void main(final String[] args) throws Exception {
		for (String baseLearner: WekaUtil.getBasicLearners()) {

			if (!baseLearner.toLowerCase().contains("j48")) {
				continue;
			}
			System.out.println(baseLearner);

			/* prepare database for this combination */
			DefaultBaseLearnerConfigContainer container = new DefaultBaseLearnerConfigContainer("conf/dbcon-local.conf", baseLearner);
			ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getExperimentSetConfig(), container.getDatabaseHandle());
			preparer.synchronizeExperiments();
		}
	}
}
