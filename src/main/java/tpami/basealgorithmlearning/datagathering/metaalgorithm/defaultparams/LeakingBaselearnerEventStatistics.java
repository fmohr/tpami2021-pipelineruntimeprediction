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

	public LeakingBaselearnerEventStatistics(LeakingBaselearnerWrapper leakingBaselearnerWrapper) {
		hashCodeOfBaselearner = leakingBaselearnerWrapper.hashCode();
	}

	public void parseEvent(LeakingBaselearnerEvent event) {
		switch (event.getEventType()) {
		case START_CLASSIFY:
			numberOfClassifyInstanceCalls++;
			if (event.getTimestamp() < firstClassifyInstanceTimestamp) {
				firstClassifyInstanceTimestamp = event.getTimestamp();
			}
			break;
		case STOP_CLASSIFY:
			if (event.getTimestamp() > lastClassifyInstanceTimestamp) {
				lastClassifyInstanceTimestamp = event.getTimestamp();
			}
			break;
		case START_DISTRIBUTION:
			numberOfDistributionCalls++;
			if (event.getTimestamp() < firstDistributionTimestamp) {
				firstDistributionTimestamp = event.getTimestamp();
			}
			break;
		case STOP_DISTRIBUTION:
			if (event.getTimestamp() > lastDistributionTimestamp) {
				lastDistributionTimestamp = event.getTimestamp();
			}
			break;
		case START_DISTRIBUTIONS:
			numberOfDistributionSCalls++;
			if (event.getTimestamp() < firstDistributionSTimestamp) {
				firstDistributionSTimestamp = event.getTimestamp();
			}
			break;
		case STOP_DISTRIBUTIONS:
			if (event.getTimestamp() > lastDistributionSTimestamp) {
				lastDistributionSTimestamp = event.getTimestamp();
			}
			break;
		case START_BUILD_CLASSIFIER:
			numberOfBuildClassifierCalls++;
			if (event.getTimestamp() < firstBuildClassifierTimestamp) {
				firstBuildClassifierTimestamp = event.getTimestamp();
			}
			break;

		case STOP_BUILD_CLASSIFIER:
			if (event.getTimestamp() > lastBuildClassifierTimestamp) {
				lastBuildClassifierTimestamp = event.getTimestamp();
			}
			break;

		case START_METAFEATURE_COMPUTATION:
			numberOfMetafeatureComputationCalls++;
			if (event.getTimestamp() < firstMetafeatureTimestamp) {
				firstMetafeatureTimestamp = event.getTimestamp();
			}
			break;

		case STOP_METAFEATURE_COMPUTATION:
			if (event.getTimestamp() > lastMetafeatureTimestamp) {
				lastMetafeatureTimestamp = event.getTimestamp();
			}
			datasetMetafeatures = event.getDatasetMetafeatures();
			break;
		case EXCEPTION:
			exception = event.getException();
		}
	}

	public int getNumberOfDistributionCalls() {
		return numberOfDistributionCalls;
	}

	public int getNumberOfDistributionSCalls() {
		return numberOfDistributionSCalls;
	}

	public int getNumberOfClassifyInstanceCalls() {
		return numberOfClassifyInstanceCalls;
	}

	public int getNumberOfBuildClassifierCalls() {
		return numberOfBuildClassifierCalls;
	}

	public int getNumberOfMetafeatureComputationCalls() {
		return numberOfMetafeatureComputationCalls;
	}

	public long getHashCodeOfBaselearner() {
		return hashCodeOfBaselearner;
	}

	public long getFirstDistributionTimestamp() {
		return firstDistributionTimestamp;
	}

	public long getLastDistributionTimestamp() {
		return lastDistributionTimestamp;
	}

	public long getFirstDistributionSTimestamp() {
		return firstDistributionSTimestamp;
	}

	public long getLastDistributionSTimestamp() {
		return lastDistributionSTimestamp;
	}

	public long getFirstClassifyInstanceTimestamp() {
		return firstClassifyInstanceTimestamp;
	}

	public long getLastClassifyInstanceTimestamp() {
		return lastClassifyInstanceTimestamp;
	}

	public long getFirstBuildClassifierTimestamp() {
		return firstBuildClassifierTimestamp;
	}

	public long getLastBuildClassifierTimestamp() {
		return lastBuildClassifierTimestamp;
	}

	public long getFirstMetafeatureTimestamp() {
		return firstMetafeatureTimestamp;
	}

	public long getLastMetafeatureTimestamp() {
		return lastMetafeatureTimestamp;
	}

	public Map<String, Object> getDatasetMetafeatures() {
		return datasetMetafeatures;
	}

	public Map<String, Object> getAsInsertableMap() {
		Map<String, Object> insertableMap = new HashMap<>();

		insertableMap.put("hashCodeOfBaselearner", hashCodeOfBaselearner);
		insertableMap.put("numberOfDistributionCalls", numberOfDistributionCalls);
		insertableMap.put("numberOfDistributionSCalls", numberOfDistributionSCalls);
		insertableMap.put("numberOfClassifyInstanceCalls", numberOfClassifyInstanceCalls);
		insertableMap.put("numberOfBuildClassifierCalls", numberOfBuildClassifierCalls);
		insertableMap.put("numberOfMetafeatureComputationCalls", numberOfMetafeatureComputationCalls);
		insertableMap.put("firstDistributionTimestamp", firstDistributionTimestamp);
		insertableMap.put("lastDistributionTimestamp", lastDistributionTimestamp);
		insertableMap.put("firstDistributionSTimestamp", firstDistributionSTimestamp);
		insertableMap.put("lastDistributionSTimestamp", lastDistributionSTimestamp);
		insertableMap.put("firstClassifyInstanceTimestamp", firstClassifyInstanceTimestamp);
		insertableMap.put("lastClassifyInstanceTimestamp", lastClassifyInstanceTimestamp);
		insertableMap.put("firstBuildClassifierTimestamp", firstBuildClassifierTimestamp);
		insertableMap.put("lastBuildClassifierTimestamp", lastBuildClassifierTimestamp);
		insertableMap.put("firstMetafeatureTimestamp", firstMetafeatureTimestamp);
		insertableMap.put("lastMetafeatureTimestamp", lastMetafeatureTimestamp);
		insertableMap.put("datasetMetafeatures", datasetMetafeatures.toString());

		return insertableMap;
	}

	@Override
	public String toString() {
		return "LeakingBaselearnerEventStatistics [hashCodeOfBaselearner=" + hashCodeOfBaselearner + ", numberOfDistributionCalls=" + numberOfDistributionCalls + ", numberOfDistributionSCalls=" + numberOfDistributionSCalls
				+ ", numberOfClassifyInstanceCalls=" + numberOfClassifyInstanceCalls + ", numberOfBuildClassifierCalls=" + numberOfBuildClassifierCalls + ", numberOfMetafeatureComputationCalls=" + numberOfMetafeatureComputationCalls
				+ ", firstDistributionTimestamp=" + firstDistributionTimestamp + ", lastDistributionTimestamp=" + lastDistributionTimestamp + ", firstDistributionSTimestamp=" + firstDistributionSTimestamp + ", lastDistributionSTimestamp="
				+ lastDistributionSTimestamp + ", firstClassifyInstanceTimestamp=" + firstClassifyInstanceTimestamp + ", lastClassifyInstanceTimestamp=" + lastClassifyInstanceTimestamp + ", firstBuildClassifierTimestamp="
				+ firstBuildClassifierTimestamp + ", lastBuildClassifierTimestamp=" + lastBuildClassifierTimestamp + ", firstMetafeatureTimestamp=" + firstMetafeatureTimestamp + ", lastMetafeatureTimestamp=" + lastMetafeatureTimestamp
				+ ", datasetMetafeatures=" + datasetMetafeatures + ", exception=" + exception + "]";
	}

}
