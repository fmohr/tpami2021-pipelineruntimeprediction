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
	@DefaultValue("500")
	public int getMaxSamples();

	@Key("calibration.seed")
	@DefaultValue("42")
	public long getSeed();

	@Key("calibration.evaltime_filter.enable")
	@DefaultValue("true")
	public boolean getEnableBaselineEvaluationTimeFilter();

	@Key("calibration.evaltime_filter.min_baseline_time")
	@DefaultValue("1")
	public Integer getMinBaselineEvaluationTime();

	@Key("calibration.evaltime_filter.max_baseline_time")
	@DefaultValue("200")
	public Integer getMaxBaselineEvaluationTime();

	@Key("calibration.config.mode")
	@DefaultValue(E_MODE_CROSS_PRODUCT)
	public String getCalibrationConfigMode();

	@Key("calibration.config.fitSizes")
	@DefaultValue("16,50,75,100,500")
	public List<String> getCalibrationConfigFitSizes();

	@Key("calibration.config.datasetIDs")
	@DefaultValue("183,1457,40975,40927,31,4136,1481,41065,1501,181")
	public List<String> getCalibrationConfigDatasetIDs();

	/* Dataset Files and Directories */
	@Key("data.base.evalruntime")
	@DefaultValue("python/data/runtimes_all_default_nooutliers.csv")
	public File getBasicEvaluationRuntimeFile();

}
