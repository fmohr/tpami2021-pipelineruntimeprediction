package tpami.safeguard;

import java.io.File;
import java.util.List;

import ai.libs.jaicore.basic.IOwnerBasedConfig;

public interface ISimpleHierarchicalRFSafeGuardConfig extends IOwnerBasedConfig {

	public static final String K_CPUS = "cpus";

	@Key(K_CPUS)
	@DefaultValue("4")
	public int getNumCPUs();

	/* Dataset Labels */
	@Key("label.openmlid")
	@DefaultValue("openmlid")
	public String getLabelForDatasetID();

	@Key("label.algorithm")
	@DefaultValue("algorithm")
	public String getLabelForAlgorithm();

	@Key("label.applicationtime")
	@DefaultValue("applicationtime")
	public String getLabelForApplicationTime();

	@Key("label.applicationsize")
	@DefaultValue("applicationsize")
	public String getLabelForApplicationSize();

	/* Dataset Files and Directories */
	@Key("data.base.evalruntime")
	@DefaultValue("python/data/runtimes_all_default_nooutliers.csv")
	public File getBasicEvaluationRuntimeFile();

	@Key("data.base.parameterized")
	@DefaultValue("python/data/parameterized/")
	public File getBasicParameterizedDirectory();

	@Key("data.base.parameterized.filename_template")
	@DefaultValue("runtimes_%s_parametrized_nooutliers.csv")
	public String getParameterizedFileNameTemplate();

	@Key("data.meta.parameterized")
	@DefaultValue("python/data/metalearner/")
	public File getMetaLearnerDirectory();

	/* Flags which parts of the safe guard to build */
	@Key("build.calibration")
	@DefaultValue("false")
	public boolean getPerformCalibration();

	@Key("build.base_components")
	@DefaultValue("true")
	public boolean getBuildBaseComponents();

	@Key("build.preprocesor_effects")
	@DefaultValue("true")
	public boolean getBuildPreprocessorEffects();

	@Key("build.meta_learner")
	@DefaultValue("true")
	public boolean getBuildMetaLearnerComponents();

	/* Debugging Configs */
	@Key("debug.test_pipeline_only")
	@DefaultValue("false")
	public boolean debuggingTestPipelineOnly();

	@Key("debug.test_pipeline_components")
	@DefaultValue("bf/cfssubseteval,weka.classifiers.trees.J48,weka.classifiers.meta.AdaBoostM1")
	public List<String> debuggingTestPipelineComponents();
}
