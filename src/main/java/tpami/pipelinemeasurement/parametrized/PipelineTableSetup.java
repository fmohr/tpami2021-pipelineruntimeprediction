package tpami.pipelinemeasurement.parametrized;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.experiments.IExperimentSetConfig;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;

public class PipelineTableSetup {

	public static void main(final String[] args) throws Exception {

		/* prepare database for this combination */
		IExperimentSetConfig configExp = (IExperimentSetConfig)ConfigFactory.create(IExperimentSetConfig.class).loadPropertiesFromFile(new File("conf/experiments/parametrized/pipelines.conf"));
		IDatabaseConfig configDB = (IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File(args[0]));
		configDB.setProperty(IDatabaseConfig.DB_TABLE, "pipelines");
		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(configExp, new ExperimenterMySQLHandle(configDB));
		preparer.synchronizeExperiments();
	}
}
