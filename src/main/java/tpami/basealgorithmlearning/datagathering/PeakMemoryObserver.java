package tpami.basealgorithmlearning.datagathering;

import java.util.concurrent.atomic.AtomicLong;

public class PeakMemoryObserver extends Thread {

	private AtomicLong memoryPeak = new AtomicLong();

	@Override
	public void run() {
		while (!Thread.interrupted()) {
			long currentMemoryConsumption = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
			this.memoryPeak.set(Math.max(this.memoryPeak.get(), currentMemoryConsumption));
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	public long getMaxMemoryConsumptionObserved() {
		return this.memoryPeak.get();
	}

	public void reset() {
		this.memoryPeak.set(0);
	}
}
