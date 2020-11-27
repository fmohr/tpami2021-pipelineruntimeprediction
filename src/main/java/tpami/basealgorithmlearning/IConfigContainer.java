package tpami.basealgorithmlearning;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentSetConfig;

public interface IConfigContainer {

	public IExperimentDatabaseHandle getDatabaseHandle();

	public IDatabaseAdapter getAdapter();

	public IExperimentSetConfig getExperimentSetConfig();
}
