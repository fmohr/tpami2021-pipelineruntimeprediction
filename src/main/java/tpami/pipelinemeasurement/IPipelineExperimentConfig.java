package tpami.pipelinemeasurement;

import java.util.List;

import ai.libs.jaicore.experiments.IExperimentSetConfig;

public interface IPipelineExperimentConfig extends IExperimentSetConfig {
	public static final String KEY_OPENMLID = "openmlid";
	public static final String KEY_DATAPOINTS = "datapoints";
	public static final String KEY_SEED = "seed";
	public static final String KEY_PREPROCESSOR = "preprocessor";
	public static final String KEY_BASE_LEARNER = "baselearner";
	public static final String KEY_META_LEARNER = "metalearner";

	@Key(KEY_OPENMLID)
	public List<Integer> openMLIDs();

	@Key(KEY_DATAPOINTS)
	public List<Integer> datapoints();

	@Key(KEY_SEED)
	public List<Integer> seeds();

	@Key(KEY_PREPROCESSOR)
	public String getPreprocessor();

	@Key(KEY_BASE_LEARNER)
	public String getBaseLearner();

	@Key(KEY_META_LEARNER)
	public String getMetaLearner();
}
