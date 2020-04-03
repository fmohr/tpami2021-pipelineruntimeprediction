package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.ml.weka.WekaUtil;

public class DefaultMetaLearnerTableSetup {

	public static void main(final String[] args) throws Exception {
		for (String metaLearner : WekaUtil.getMetaLearners()) {

			System.out.println(metaLearner);

			/* prepare database for this combination */
			DefaultMetaLearnerConfigContainer container = new DefaultMetaLearnerConfigContainer("conf/dbcon-local.conf", metaLearner);
			ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
			preparer.setLoggerName("example");
			preparer.synchronizeExperiments();
		}
	}
}
