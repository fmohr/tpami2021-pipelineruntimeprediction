package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;

public class DefaultMetaLearnerConfigContainer {
	private final IDefaultMetaLearnerExperimentConfig config;
	private final IExperimentDatabaseHandle databaseHandle;
	private final IDatabaseAdapter adapter;

	public DefaultMetaLearnerConfigContainer(final String databaseConfigFile, final String metaLearnerName) throws Exception {

		/* get experiment configuration */
		final Class<?> classifierClass = Class.forName(metaLearnerName);
		this.config = ConfigFactory.create(IDefaultMetaLearnerExperimentConfig.class);
		String classifierWorkingName = classifierClass.getSimpleName().toLowerCase();
		this.config.loadPropertiesFromFile(new File("conf/experiments/defaultparams/metalearner.conf"));

		/* setup database connection */
		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		dbConfig.loadPropertiesFromFile(new File(databaseConfigFile));
		this.adapter = DatabaseAdapterFactory.get(dbConfig);
		this.databaseHandle = new ExperimenterMySQLHandle(this.adapter, "evaluations_metaclassifiers_" + classifierWorkingName);
	}

	public IDefaultMetaLearnerExperimentConfig getConfig() {
		return this.config;
	}

	public IExperimentDatabaseHandle getDatabaseHandle() {
		return this.databaseHandle;
	}

	public IDatabaseAdapter getAdapter() {
		return this.adapter;
	}
}
