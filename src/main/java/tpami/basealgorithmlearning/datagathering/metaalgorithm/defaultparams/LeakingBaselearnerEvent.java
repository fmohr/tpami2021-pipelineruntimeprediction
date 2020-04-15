package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.Map;

import org.apache.commons.lang3.exception.ExceptionUtils;

import tpami.basealgorithmlearning.datagathering.DataGatheringUtil;

public class LeakingBaselearnerEvent {

	private final boolean metaLearnerIsTrained;
	private ELeakingBaselearnerEventType eventType;
	private long timestamp;
	private Map<String, Object> datasetMetafeatures;
	private LeakingBaselearnerWrapper leakingBaselearnerWrapper;
	private Exception exception = null;

	public LeakingBaselearnerEvent(final ELeakingBaselearnerEventType eventType, final LeakingBaselearnerWrapper leakingBaselearnerWrapper, final boolean metaLearnerIsTrained) {
		this.eventType = eventType;
		this.timestamp = DataGatheringUtil.getCurrentFormattedTimestamp();
		this.leakingBaselearnerWrapper = leakingBaselearnerWrapper;
		this.metaLearnerIsTrained = metaLearnerIsTrained;
	}

	public LeakingBaselearnerEvent(final Map<String, Object> datasetMetafeatures, final LeakingBaselearnerWrapper leakingBaselearnerWrapper, final boolean metaLearnerIsTrained) {
		this(ELeakingBaselearnerEventType.STOP_METAFEATURE_COMPUTATION, leakingBaselearnerWrapper, metaLearnerIsTrained);
		this.datasetMetafeatures = datasetMetafeatures;
	}

	public LeakingBaselearnerEvent(final Exception exception, final boolean duringTraining) {
		this.metaLearnerIsTrained = duringTraining;
		this.exception = exception;
		this.eventType = ELeakingBaselearnerEventType.EXCEPTION;
	}

	public ELeakingBaselearnerEventType getEventType() {
		return this.eventType;
	}

	public long getTimestamp() {
		return this.timestamp;
	}

	public Map<String, Object> getDatasetMetafeatures() {
		return this.datasetMetafeatures;
	}

	public LeakingBaselearnerWrapper getLeakingBaselearnerWrapper() {
		return this.leakingBaselearnerWrapper;
	}

	public Exception getException() {
		return this.exception;
	}

	public String getStacktrace() {
		return ExceptionUtils.getStackTrace(this.exception);
	}

	public boolean isMetaLearnerTrained() {
		return this.metaLearnerIsTrained;
	}
}
