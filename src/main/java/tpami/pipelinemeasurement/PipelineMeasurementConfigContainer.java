package tpami.pipelinemeasurement;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;

public class PipelineMeasurementConfigContainer {
	private final IPipelineExperimentConfig config;
	private final IExperimentDatabaseHandle databaseHandle;
	private final IDatabaseAdapter adapter;

	public PipelineMeasurementConfigContainer(final String databaseConfigFile) throws Exception {

		/* get experiment configuration */
		this.config = ConfigFactory.create(IPipelineExperimentConfig.class);
		this.config.loadPropertiesFromFile(new File("conf/experiments/defaultparams/pipelines.conf"));

		/* setup database connection */
		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		dbConfig.loadPropertiesFromFile(new File(databaseConfigFile));
		this.adapter = DatabaseAdapterFactory.get(dbConfig);
		this.databaseHandle = new ExperimenterMySQLHandle(this.adapter, "evaluations_pipelines");
	}

	public IPipelineExperimentConfig getConfig() {
		return this.config;
	}

	public IExperimentDatabaseHandle getDatabaseHandle() {
		return this.databaseHandle;
	}

	public IDatabaseAdapter getAdapter() {
		return this.adapter;
	}
}
