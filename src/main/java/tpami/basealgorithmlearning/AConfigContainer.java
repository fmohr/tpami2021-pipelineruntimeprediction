package tpami.basealgorithmlearning;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentSetConfig;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;

public class AConfigContainer implements IConfigContainer {

	protected final IExperimentDatabaseHandle databaseHandle;
	protected final IDatabaseAdapter adapter;
	protected final IExperimentSetConfig config;

	public AConfigContainer(final String experimentSetConfigFile, final String databaseConfigFile, final String experimentTable) {

		/* setup database connection */
		this((IExperimentSetConfig)ConfigFactory.create(IExperimentSetConfig.class).loadPropertiesFromFile(new File(experimentSetConfigFile)), new SQLAdapter((IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File(databaseConfigFile))), experimentTable);
	}

	AConfigContainer(final IExperimentSetConfig experimentConfig, final IDatabaseAdapter adapter, final String experimentTable) {
		this(experimentConfig, new ExperimenterMySQLHandle(adapter, experimentTable), adapter);
	}

	AConfigContainer(final IExperimentSetConfig experimentConfig, final IExperimentDatabaseHandle databaseHandle, final IDatabaseAdapter adapter) {
		super();
		this.config = experimentConfig;
		this.databaseHandle = databaseHandle;
		this.adapter = adapter;
	}

	@Override
	public IExperimentDatabaseHandle getDatabaseHandle() {
		return this.databaseHandle;
	}

	@Override
	public IDatabaseAdapter getAdapter() {
		return this.adapter;
	}

	@Override
	public IExperimentSetConfig getExperimentSetConfig() {
		return this.config;
	}
}
