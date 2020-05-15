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

	/* Configs for reading basic component datasets */
	@Key("data.runtime.base")
	@DefaultValue("bayesnet,decisionstump,decisiontable,ibk,j48,jrip,kstar,lmt,logistic,multilayerperceptron,naivebayes,naivebayesmultinomial,oner,randomforest,randomtree,reptree,simplelogistic,smo,votedperceptron,zeror,bestfirst_cfssubseteval,greedystepwise_cfssubseteval,ranker_correlationattributeeval,ranker_gainratioattributeeval,ranker_infogainattributeeval,ranker_onerattributeeval,ranker_principalcomponents,ranker_relieffattributeeval,ranker_symmetricaluncertattributeeval")
	public List<String> getBasicComponentsForRuntime();

	@Key("data.runtime.base.dir")
	@DefaultValue("data2/runtime/")
	public File getBasicComponentsForDefaultRuntimeDirectory();

	@Key("data.runtime.base.filepattern.default")
	@DefaultValue("runtimes_%s_default.csv")
	public String getBasicComponentsForDefaultRuntimeFilePattern(final String name);

	@Key("data.runtime.base.filepattern.default")
	@DefaultValue("runtimes_%s_parametrized.csv")
	public String getBasicComponentsForParamRuntimeFilePattern(final String name);

	/* Configs for reading metalearner datasets */
	@Key("data.transform.metalearner")
	@DefaultValue("adaboostm1,bagging,logitboost,randomcommittee,randomsubspace")
	public List<String> getMetaLearnerTransformEffect();

	@Key("data.transform.metalearner.dir")
	@DefaultValue("data2/transform/metalearner/")
	public File getMetaLearnerTransformEffectDirectory();

	@Key("data.transform.metalearner.filepattern")
	@DefaultValue("metalearner_parametereffects_%s.csv")
	public String getMetaLearnersForTransformEffectFilePattern(final String name);

	/* Configs for reading preprocessor datasets */
	@Key("data.transform.preprocessing")
	@DefaultValue("bestfirst_cfssubseteval,greedystepwise_cfssubseteval,ranker_correlationattributeeval,ranker_gainratioattributeeval,ranker_infogainattributeeval,ranker_onerattributeeval,ranker_principalcomponents,ranker_relieffattributeeval,ranker_symmetricaluncertattributeeval")
	public List<String> getPreprocessorsForTransformEffect();

	@Key("data.transform.preprocessing.dir")
	@DefaultValue("data2/transform/preprocessing/")
	public File getPreprocessorsForTransformEffectDirectory();

	@Key("data.transform.preprocessing.filepattern")
	@DefaultValue("%s.csv")
	public String getPreprocessorsForTransformEffectFilePattern(String name);

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

	/* Flags which parts of the safe guard to build */
	@Key("build.calibration")
	@DefaultValue("true")
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
	@DefaultValue("greedystepwise_cfssubseteval,j48,adaboostm1")
	public List<String> debuggingTestPipelineComponents();
}
