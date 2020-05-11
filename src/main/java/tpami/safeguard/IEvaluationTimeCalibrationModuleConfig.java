package tpami.safeguard;

import java.io.File;
import java.util.List;

import org.aeonbits.owner.Config.Sources;

import ai.libs.jaicore.basic.IOwnerBasedConfig;

@Sources({ "file:conf/evaluationTimeCalibrationModule.conf" })
public interface IEvaluationTimeCalibrationModuleConfig extends IOwnerBasedConfig {

	public static final String K_CPUS = "cpus";

	public static final String E_MODE_CROSS_PRODUCT = "crossProduct";
	public static final String E_MODE_PAIR_WISE = "pairWise";

	@Key(K_CPUS)
	@DefaultValue("4")
	public int getNumCPUs();

	@Key("calibration.max_samples")
	@DefaultValue("100")
	public int getMaxSamples();

	@Key("calibration.seed")
	@DefaultValue("42")
	public long getSeed();

	@Key("calibration.evaltime_filter.enable")
	@DefaultValue("true")
	public boolean getEnableBaselineEvaluationTimeFilter();

	@Key("calibration.evaltime_filter.min_baseline_time")
	@DefaultValue("10")
	public Integer getMinBaselineEvaluationTime();

	@Key("calibration.evaltime_filter.max_baseline_time")
	@DefaultValue("300")
	public Integer getMaxBaselineEvaluationTime();

	@Key("calibration.config.mode")
	@DefaultValue(E_MODE_CROSS_PRODUCT)
	public String getCalibrationConfigMode();

	@Key("calibration.config.fitSizes")
	@DefaultValue("")
	public List<String> getCalibrationConfigFitSizes();

	@Key("calibration.config.datasetIDs")
	@DefaultValue("")
	public List<String> getCalibrationConfigDatasetIDs();

	/* Dataset Files and Directories */
	@Key("data.base.evalruntime")
	@DefaultValue("python/data/runtimes_all_default_nooutliers.csv")
	public File getBasicEvaluationRuntimeFile();

}
