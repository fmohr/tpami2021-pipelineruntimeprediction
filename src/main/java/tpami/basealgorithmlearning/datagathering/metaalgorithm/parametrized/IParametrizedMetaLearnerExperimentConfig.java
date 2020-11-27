package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized;

import java.util.List;

import ai.libs.jaicore.experiments.IExperimentSetConfig;

public interface IParametrizedMetaLearnerExperimentConfig extends IExperimentSetConfig {
	public static final String KEY_OPENMLID = "openmlid";
	public static final String KEY_DATAPOINTS = "datapoints";
	public static final String KEY_SEED = "seed";
	public static final String KEY_ALGORITHMOPTIONS = "algorithmoptions";
	public static final String BASE_LEARNER = "baselearner";

	@Key(KEY_OPENMLID)
	public List<Integer> openMLIDs();

	@Key(KEY_DATAPOINTS)
	public List<Integer> datapoints();

	@Key(KEY_SEED)
	public List<Integer> seeds();

	@Key(KEY_ALGORITHMOPTIONS)
	public String getAlgorithmOptions();

	@Key(BASE_LEARNER)
	public String getBaseLearner();
}
