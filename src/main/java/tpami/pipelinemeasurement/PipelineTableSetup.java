package tpami.pipelinemeasurement;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;

public class PipelineTableSetup {

	public static void main(final String[] args) throws Exception {

		/* prepare database for this combination */
		PipelineMeasurementConfigContainer container = new PipelineMeasurementConfigContainer("conf/dbcon-local.conf");
		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
		preparer.setLoggerName("example");
		preparer.synchronizeExperiments();
	}
}
