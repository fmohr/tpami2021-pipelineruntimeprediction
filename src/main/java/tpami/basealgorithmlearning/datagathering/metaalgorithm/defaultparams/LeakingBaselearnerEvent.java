package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.Map;

import org.apache.commons.lang3.exception.ExceptionUtils;

import tpami.basealgorithmlearning.datagathering.DataGatheringUtil;

public class LeakingBaselearnerEvent {

	private ELeakingBaselearnerEventType eventType;
	private long timestamp;
	private Map<String, Object> datasetMetafeatures;
	private LeakingBaselearnerWrapper leakingBaselearnerWrapper;
	private Exception exception = null;

	public LeakingBaselearnerEvent(ELeakingBaselearnerEventType eventType, LeakingBaselearnerWrapper leakingBaselearnerWrapper) {
		this.eventType = eventType;
		this.timestamp = DataGatheringUtil.getCurrentFormattedTimestamp();
		this.leakingBaselearnerWrapper = leakingBaselearnerWrapper;
	}

	public LeakingBaselearnerEvent(Map<String, Object> datasetMetafeatures, LeakingBaselearnerWrapper leakingBaselearnerWrapper) {
		this(ELeakingBaselearnerEventType.STOP_METAFEATURE_COMPUTATION, leakingBaselearnerWrapper);
		this.datasetMetafeatures = datasetMetafeatures;
	}

	public LeakingBaselearnerEvent(Exception exception) {
		this.exception = exception;
		this.eventType = ELeakingBaselearnerEventType.EXCEPTION;
	}

	public ELeakingBaselearnerEventType getEventType() {
		return eventType;
	}

	public long getTimestamp() {
		return timestamp;
	}

	public Map<String, Object> getDatasetMetafeatures() {
		return datasetMetafeatures;
	}

	public LeakingBaselearnerWrapper getLeakingBaselearnerWrapper() {
		return leakingBaselearnerWrapper;
	}

	public Exception getException() {
		return exception;
	}

	public String getStacktrace() {
		return ExceptionUtils.getStackTrace(exception);
	}

}
