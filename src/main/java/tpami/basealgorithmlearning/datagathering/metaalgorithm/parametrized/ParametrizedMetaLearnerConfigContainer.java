package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;

public class ParametrizedMetaLearnerConfigContainer {
	private final IParametrizedMetaLearnerExperimentConfig config;
	private final IExperimentDatabaseHandle databaseHandle;
	private final IDatabaseAdapter adapter;

	public ParametrizedMetaLearnerConfigContainer(final String databaseConfigFile, final String metaLearnerName) throws Exception {

		/* get experiment configuration */
		final Class<?> classifierClass = Class.forName(metaLearnerName);
		this.config = ConfigFactory.create(IParametrizedMetaLearnerExperimentConfig.class);
		String classifierWorkingName = classifierClass.getSimpleName().toLowerCase();
		this.config.loadPropertiesFromFile(new File("conf/experiments/parametrized/metalearner-" + classifierWorkingName + ".conf"));

		/* setup database connection */
		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		dbConfig.loadPropertiesFromFile(new File(databaseConfigFile));
		this.adapter = DatabaseAdapterFactory.get(dbConfig);
		this.databaseHandle = new ExperimenterMySQLHandle(this.adapter, "evaluations_metaclassifiers_" + classifierWorkingName + "_configured");
	}

	public IParametrizedMetaLearnerExperimentConfig getConfig() {
		return this.config;
	}

	public IExperimentDatabaseHandle getDatabaseHandle() {
		return this.databaseHandle;
	}

	public IDatabaseAdapter getAdapter() {
		return this.adapter;
	}
}
