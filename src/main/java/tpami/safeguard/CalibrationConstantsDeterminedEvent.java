package tpami.safeguard;

import org.api4.java.common.event.IEvent;

public class CalibrationConstantsDeterminedEvent implements IEvent {

	private long timestamp;

	private final double cInduction;
	private final double cInference;

	public CalibrationConstantsDeterminedEvent(final double cInduction, final double cInference) {
		this.timestamp = System.currentTimeMillis();
		this.cInduction = cInduction;
		this.cInference = cInference;
	}

	public double getCInduction() {
		return this.cInduction;
	}

	public double getCInference() {
		return this.cInference;
	}

	@Override
	public long getTimestamp() {
		return this.timestamp;
	}

}
