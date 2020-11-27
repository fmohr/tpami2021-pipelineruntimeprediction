package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import weka.classifiers.lazy.LWL;

public class ParametrizedMetaLearnerTableSetup {

	public static void main(final String[] args) throws Exception {
		//		for (String metaLearner : WekaUtil.getMetaLearners()) {
		//
		//			if (metaLearner.contains("Additive") || metaLearner.contains("AttributeSelected") || metaLearner.contains("ClassificationViaRegression") || metaLearner.contains("MultiClass") || metaLearner.contains("Stacking")  || metaLearner.contains("Vote")) {
		//				continue;
		//			}
		//
		//			if (!metaLearner.contains("Logit")) {
		//				continue;
		//			}
		//
		//			/* prepare database for this combination */
		//			ParametrizedMetaLearnerConfigContainer container = new ParametrizedMetaLearnerConfigContainer("conf/dbcon-local.conf", metaLearner);
		//			ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
		//			preparer.setLoggerName("example");
		//			preparer.synchronizeExperiments();
		//		}

		String metaLearner = LWL.class.getName();
		ParametrizedMetaLearnerConfigContainer container = new ParametrizedMetaLearnerConfigContainer("conf/dbcon-local.conf", metaLearner);
		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
		preparer.setLoggerName("example");
		preparer.synchronizeExperiments();
	}
}
