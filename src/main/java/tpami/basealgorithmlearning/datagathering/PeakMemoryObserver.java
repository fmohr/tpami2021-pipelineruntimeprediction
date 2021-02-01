package tpami.basealgorithmlearning.datagathering;

import java.util.concurrent.atomic.AtomicLong;

import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PeakMemoryObserver extends Thread implements ILoggingCustomizable{

	private int logCycleInSeconds = 10;
	private Logger logger = LoggerFactory.getLogger(PeakMemoryObserver.class);
	private AtomicLong memoryPeak = new AtomicLong();
	private boolean stopped = false;

	public PeakMemoryObserver() {
		super("Memory Peak Observer");
	}

	@Override
	public void run() {
		final Runtime r = Runtime.getRuntime();
		long timeForNextLogMessage = System.currentTimeMillis();
		long maxMemory = r.maxMemory();
		while (!this.stopped && !Thread.interrupted()) {
			long totalMemory = r.totalMemory();
			long freeMemory = r.freeMemory();
			long currentMemoryConsumption = (totalMemory - freeMemory);
			long newPeak = Math.max(this.memoryPeak.get(), currentMemoryConsumption);
			this.memoryPeak.set(newPeak);
			long now = System.currentTimeMillis();
			if (now >= timeForNextLogMessage) {
				this.logger.info("Memory consumption stats. Current Consumption: {}MB. Peak Consumption: {}MB. Currently available: {}MB", currentMemoryConsumption / 1024 / 1024, newPeak / 1024 / 1024, (maxMemory - currentMemoryConsumption) / 1024 / 1024);
				timeForNextLogMessage = now + this.logCycleInSeconds * 1000;
			}
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		this.logger.info("Memory Peak Observer stops.");
	}

	public long getMaxMemoryConsumptionObserved() {
		return this.memoryPeak.get();
	}

	public void reset() {
		this.memoryPeak.set(0);
	}

	public void cancel() {
		this.stopped = true;
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}
}
