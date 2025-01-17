package tpami.basealgorithmlearning.datagathering;

import java.util.List;

import ai.libs.jaicore.experiments.IExperimentSetConfig;

public interface ILearnerExperimentConfig extends IExperimentSetConfig {
	public static final String KEY_OPENMLID = "openmlid";
	public static final String KEY_DATAPOINTS = "datapoints";
	public static final String KEY_SEED = "seed";

	@Key(KEY_OPENMLID)
	public List<Integer> openMLIDs();

	@Key(KEY_DATAPOINTS)
	public List<Integer> datapoints();

	@Key(KEY_SEED)
	public List<Integer> seeds();
}
