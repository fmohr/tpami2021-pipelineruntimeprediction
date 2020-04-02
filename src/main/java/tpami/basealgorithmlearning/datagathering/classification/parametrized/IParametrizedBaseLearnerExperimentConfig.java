package tpami.basealgorithmlearning.datagathering.classification.parametrized;

import java.util.List;

import ai.libs.jaicore.experiments.IExperimentSetConfig;

public interface IParametrizedBaseLearnerExperimentConfig extends IExperimentSetConfig {
	public static final String KEY_OPENMLID = "openmlid";
	public static final String KEY_DATAPOINTS = "datapoints";
	public static final String KEY_OPTIONS = "algorithmoptions";
	public static final String KEY_SEED = "seed";

	@Key(KEY_OPENMLID)
	public List<Integer> openMLIDs();

	@Key(KEY_DATAPOINTS)
	public List<Integer> datapoints();

	@Key(KEY_OPTIONS)
	public List<String> options();

	@Key(KEY_SEED)
	public List<Integer> seeds();
}
