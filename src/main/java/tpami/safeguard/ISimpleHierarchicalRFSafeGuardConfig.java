package tpami.safeguard;

import java.io.File;
import java.util.List;

import ai.libs.jaicore.basic.IOwnerBasedConfig;

public interface ISimpleHierarchicalRFSafeGuardConfig extends IOwnerBasedConfig {

	public static final String FILE_PATTERN_BASIC_DEF = "runtimes_%s_default.csv";
	public static final String FILE_PATTERN_BASIC_PAR = "runtimes_%s_parametrized.csv";
	public static final String FILE_PATTERN_PREPROCESSOR = "%s.csv";
	public static final String FILE_PATTERN_METALEARNER = "metalearner_parametereffects_%s.csv";

	public static final String K_CPUS = "cpus";

	public static final String K_ENABLE_CALIBRATION = "build.calibration";
	public static final String K_ENABLE_BASE_COMPONENTS = "build.base_components";
	public static final String K_ENABLE_PREPROCESSOR_EFFECTS = "build.preprocesor_effects";
	public static final String K_ENABLE_META_LEARNER_EFFECTS = "build.meta_learner";

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

	/* Configs for reading datasets */
	@Key("data.runtime.base.dir")
	@DefaultValue("data/runtime/")
	public File getBasicComponentsForDefaultRuntimeDirectory();

	@Key("data.transform.metalearner.dir")
	@DefaultValue("data/transform/metalearner/")
	public File getMetaLearnerTransformEffectDirectory();

	@Key("data.transform.preprocessing.dir")
	@DefaultValue("data/transform/preprocessing/")
	public File getPreprocessorsForTransformEffectDirectory();

	@Key("data.runtime.base")
	@DefaultValue("bayesnet,decisionstump,decisiontable,ibk,j48,jrip,kstar,lmt,logistic,multilayerperceptron,naivebayes,naivebayesmultinomial,oner,randomforest,randomtree,reptree,simplelogistic,smo,votedperceptron,zeror,bestfirst_cfssubseteval,greedystepwise_cfssubseteval,ranker_correlationattributeeval,ranker_gainratioattributeeval,ranker_infogainattributeeval,ranker_onerattributeeval,ranker_principalcomponents,ranker_relieffattributeeval,ranker_symmetricaluncertattributeeval")
	public List<String> getBasicComponentsForRuntime();

	@Key("data.transform.metalearner")
	@DefaultValue("adaboostm1,bagging,logitboost,randomcommittee,randomsubspace")
	public List<String> getMetaLearnerTransformEffect();

	@Key("data.transform.preprocessing")
	@DefaultValue("bestfirst_cfssubseteval,greedystepwise_cfssubseteval,ranker_correlationattributeeval,ranker_gainratioattributeeval,ranker_infogainattributeeval,ranker_onerattributeeval,ranker_principalcomponents,ranker_relieffattributeeval,ranker_symmetricaluncertattributeeval")
	public List<String> getPreprocessorsForTransformEffect();

	/* Flags which parts of the safe guard to build */
	@Key(K_ENABLE_CALIBRATION)
	@DefaultValue("false")
	public boolean getPerformCalibration();

	@Key(K_ENABLE_BASE_COMPONENTS)
	@DefaultValue("true")
	public boolean getBuildBaseComponents();

	@Key(K_ENABLE_PREPROCESSOR_EFFECTS)
	@DefaultValue("true")
	public boolean getBuildPreprocessorEffects();

	@Key(K_ENABLE_META_LEARNER_EFFECTS)
	@DefaultValue("true")
	public boolean getBuildMetaLearnerComponents();

	/* Debugging Configs */
	@Key("debug.test_pipeline_only")
	@DefaultValue("false")
	public boolean debuggingTestPipelineOnly();

	@Key("debug.test_pipeline_components")
	@DefaultValue("greedystepwise_cfssubseteval,j48,adaboostm1")
	public List<String> debuggingTestPipelineComponents();
}
