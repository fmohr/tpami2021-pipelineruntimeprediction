package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

public class LeakingBaselearnerEvent {

	private ELeakingBaselearnerEventType eventType;
	private long durationInMilliseconds = -1;

	public LeakingBaselearnerEvent(ELeakingBaselearnerEventType eventType) {
		this.eventType = eventType;
	}

	public LeakingBaselearnerEvent(ELeakingBaselearnerEventType eventType, long durationInMilliseconds) {
		this.eventType = eventType;
		this.durationInMilliseconds = durationInMilliseconds;
	}

	public ELeakingBaselearnerEventType getEventType() {
		return eventType;
	}

	public boolean isSimpleCountEvent() {
		return durationInMilliseconds < 0;
	}

	public long getDurationInMilliseconds() {
		return durationInMilliseconds;
	}

}
