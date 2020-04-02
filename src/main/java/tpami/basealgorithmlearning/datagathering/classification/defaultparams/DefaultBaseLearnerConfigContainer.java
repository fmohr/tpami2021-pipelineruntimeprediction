package tpami.basealgorithmlearning.datagathering.classification.defaultparams;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;

public class DefaultBaseLearnerConfigContainer {
	private final IDefaultBaseLearnerExperimentConfig config;
	private final IExperimentDatabaseHandle databaseHandle;
	private final IDatabaseAdapter adapter;

	public DefaultBaseLearnerConfigContainer(final String databaseConfigFile, final String classifierClassName) throws Exception {

		/* get experiment configuration */
		final Class<?> classifierClass = Class.forName(classifierClassName);
		this.config = ConfigFactory.create(IDefaultBaseLearnerExperimentConfig.class);
		String classifierWorkingName = classifierClass.getSimpleName().toLowerCase();
		this.config.loadPropertiesFromFile(new File("conf/experiments/defaultparams/baselearner.conf"));

		/* setup database connection */
		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		dbConfig.loadPropertiesFromFile(new File(databaseConfigFile));
		this.adapter = new SQLAdapter(dbConfig);
		this.databaseHandle = new ExperimenterMySQLHandle(this.adapter, "evaluations_classifiers_" + classifierWorkingName);
	}

	public IDefaultBaseLearnerExperimentConfig getConfig() {
		return this.config;
	}

	public IExperimentDatabaseHandle getDatabaseHandle() {
		return this.databaseHandle;
	}

	public IDatabaseAdapter getAdapter() {
		return this.adapter;
	}
}
