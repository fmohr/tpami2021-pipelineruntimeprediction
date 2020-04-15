package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.HashMap;
import java.util.Map;

public class LeakingBaselearnerEventStatistics {

	private long hashCodeOfBaselearner;

	private int numberOfDistributionCalls;
	private int numberOfDistributionSCalls;
	private int numberOfClassifyInstanceCalls;
	private int numberOfBuildClassifierCalls;
	private int numberOfMetafeatureComputationCalls;

	private long firstDistributionTimestamp = Long.MAX_VALUE;
	private long lastDistributionTimestamp = 0;

	private long firstDistributionSTimestamp = Long.MAX_VALUE;
	private long lastDistributionSTimestamp = 0;

	private long firstClassifyInstanceTimestamp = Long.MAX_VALUE;
	private long lastClassifyInstanceTimestamp = 0;

	private long firstBuildClassifierTimestamp = Long.MAX_VALUE;
	private long lastBuildClassifierTimestamp = 0;

	private long firstMetafeatureTimestamp = Long.MAX_VALUE;
	private long lastMetafeatureTimestamp = 0;

	private Map<String, Object> datasetMetafeatures;

	private Exception exception;

	public LeakingBaselearnerEventStatistics(final LeakingBaselearnerWrapper leakingBaselearnerWrapper) {
		this.hashCodeOfBaselearner = leakingBaselearnerWrapper.hashCode();
	}

	public void parseEvent(final LeakingBaselearnerEvent event) {
		switch (event.getEventType()) {
		case START_CLASSIFY:
			this.numberOfClassifyInstanceCalls++;
			if (event.getTimestamp() < this.firstClassifyInstanceTimestamp) {
				this.firstClassifyInstanceTimestamp = event.getTimestamp();
			}
			break;
		case STOP_CLASSIFY:
			if (event.getTimestamp() > this.lastClassifyInstanceTimestamp) {
				this.lastClassifyInstanceTimestamp = event.getTimestamp();
			}
			break;
		case START_DISTRIBUTION:
			this.numberOfDistributionCalls++;
			if (event.getTimestamp() < this.firstDistributionTimestamp) {
				this.firstDistributionTimestamp = event.getTimestamp();
			}
			break;
		case STOP_DISTRIBUTION:
			if (event.getTimestamp() > this.lastDistributionTimestamp) {
				this.lastDistributionTimestamp = event.getTimestamp();
			}
			break;
		case START_DISTRIBUTIONS:
			this.numberOfDistributionSCalls++;
			if (event.getTimestamp() < this.firstDistributionSTimestamp) {
				this.firstDistributionSTimestamp = event.getTimestamp();
			}
			break;
		case STOP_DISTRIBUTIONS:
			if (event.getTimestamp() > this.lastDistributionSTimestamp) {
				this.lastDistributionSTimestamp = event.getTimestamp();
			}
			break;
		case START_BUILD_CLASSIFIER:
			this.numberOfBuildClassifierCalls++;
			if (event.getTimestamp() < this.firstBuildClassifierTimestamp) {
				this.firstBuildClassifierTimestamp = event.getTimestamp();
			}
			break;

		case STOP_BUILD_CLASSIFIER:
			if (event.getTimestamp() > this.lastBuildClassifierTimestamp) {
				this.lastBuildClassifierTimestamp = event.getTimestamp();
			}
			break;

		case START_METAFEATURE_COMPUTATION:
			this.numberOfMetafeatureComputationCalls++;
			if (event.getTimestamp() < this.firstMetafeatureTimestamp) {
				this.firstMetafeatureTimestamp = event.getTimestamp();
			}
			break;

		case STOP_METAFEATURE_COMPUTATION:
			if (event.getTimestamp() > this.lastMetafeatureTimestamp) {
				this.lastMetafeatureTimestamp = event.getTimestamp();
			}
			this.datasetMetafeatures = event.getDatasetMetafeatures();
			break;
		case EXCEPTION:
			this.exception = event.getException();
		}
	}

	public int getNumberOfDistributionCalls() {
		return this.numberOfDistributionCalls;
	}

	public int getNumberOfDistributionSCalls() {
		return this.numberOfDistributionSCalls;
	}

	public int getNumberOfClassifyInstanceCalls() {
		return this.numberOfClassifyInstanceCalls;
	}

	public int getNumberOfBuildClassifierCalls() {
		return this.numberOfBuildClassifierCalls;
	}

	public int getNumberOfMetafeatureComputationCalls() {
		return this.numberOfMetafeatureComputationCalls;
	}

	public long getHashCodeOfBaselearner() {
		return this.hashCodeOfBaselearner;
	}

	public long getFirstDistributionTimestamp() {
		return this.firstDistributionTimestamp;
	}

	public long getLastDistributionTimestamp() {
		return this.lastDistributionTimestamp;
	}

	public long getFirstDistributionSTimestamp() {
		return this.firstDistributionSTimestamp;
	}

	public long getLastDistributionSTimestamp() {
		return this.lastDistributionSTimestamp;
	}

	public long getFirstClassifyInstanceTimestamp() {
		return this.firstClassifyInstanceTimestamp;
	}

	public long getLastClassifyInstanceTimestamp() {
		return this.lastClassifyInstanceTimestamp;
	}

	public long getFirstBuildClassifierTimestamp() {
		return this.firstBuildClassifierTimestamp;
	}

	public long getLastBuildClassifierTimestamp() {
		return this.lastBuildClassifierTimestamp;
	}

	public long getFirstMetafeatureTimestamp() {
		return this.firstMetafeatureTimestamp;
	}

	public long getLastMetafeatureTimestamp() {
		return this.lastMetafeatureTimestamp;
	}

	public Map<String, Object> getDatasetMetafeatures() {
		return this.datasetMetafeatures;
	}

	public Map<String, Object> getAsInsertableMap(final String suffix) {
		Map<String, Object> insertableMap = new HashMap<>();


		insertableMap.put("numberOfDistributionCalls_" + suffix, this.numberOfDistributionCalls);
		insertableMap.put("numberOfDistributionSCalls_" + suffix, this.numberOfDistributionSCalls);
		insertableMap.put("numberOfClassifyInstanceCalls_" + suffix, this.numberOfClassifyInstanceCalls);
		insertableMap.put("numberOfBuildClassifierCalls_" + suffix, this.numberOfBuildClassifierCalls);
		insertableMap.put("numberOfMetafeatureComputationCalls_" + suffix, this.numberOfMetafeatureComputationCalls);
		insertableMap.put("firstDistributionTimestamp_" + suffix, this.firstDistributionTimestamp);
		insertableMap.put("lastDistributionTimestamp_" + suffix, this.lastDistributionTimestamp);
		insertableMap.put("firstDistributionSTimestamp_" + suffix, this.firstDistributionSTimestamp);
		insertableMap.put("lastDistributionSTimestamp_" + suffix, this.lastDistributionSTimestamp);
		insertableMap.put("firstClassifyInstanceTimestamp_" + suffix, this.firstClassifyInstanceTimestamp);
		insertableMap.put("lastClassifyInstanceTimestamp_" + suffix, this.lastClassifyInstanceTimestamp);
		insertableMap.put("firstBuildClassifierTimestamp_" + suffix, this.firstBuildClassifierTimestamp);
		insertableMap.put("lastBuildClassifierTimestamp_" + suffix, this.lastBuildClassifierTimestamp);
		insertableMap.put("firstMetafeatureTimestamp_" + suffix, this.firstMetafeatureTimestamp);
		insertableMap.put("lastMetafeatureTimestamp_" + suffix, this.lastMetafeatureTimestamp);

		return insertableMap;
	}

	@Override
	public String toString() {
		return "LeakingBaselearnerEventStatistics [hashCodeOfBaselearner=" + this.hashCodeOfBaselearner + ", numberOfDistributionCalls=" + this.numberOfDistributionCalls + ", numberOfDistributionSCalls=" + this.numberOfDistributionSCalls
				+ ", numberOfClassifyInstanceCalls=" + this.numberOfClassifyInstanceCalls + ", numberOfBuildClassifierCalls=" + this.numberOfBuildClassifierCalls + ", numberOfMetafeatureComputationCalls=" + this.numberOfMetafeatureComputationCalls
				+ ", firstDistributionTimestamp=" + this.firstDistributionTimestamp + ", lastDistributionTimestamp=" + this.lastDistributionTimestamp + ", firstDistributionSTimestamp=" + this.firstDistributionSTimestamp + ", lastDistributionSTimestamp="
				+ this.lastDistributionSTimestamp + ", firstClassifyInstanceTimestamp=" + this.firstClassifyInstanceTimestamp + ", lastClassifyInstanceTimestamp=" + this.lastClassifyInstanceTimestamp + ", firstBuildClassifierTimestamp="
				+ this.firstBuildClassifierTimestamp + ", lastBuildClassifierTimestamp=" + this.lastBuildClassifierTimestamp + ", firstMetafeatureTimestamp=" + this.firstMetafeatureTimestamp + ", lastMetafeatureTimestamp=" + this.lastMetafeatureTimestamp
				+ ", datasetMetafeatures=" + this.datasetMetafeatures + ", exception=" + this.exception + "]";
	}

}
