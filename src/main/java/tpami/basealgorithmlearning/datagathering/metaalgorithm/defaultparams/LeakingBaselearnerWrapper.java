package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import com.google.common.eventbus.EventBus;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LeakingBaselearnerWrapper extends AbstractClassifier implements Sourcable {

	private EventBus eventBus;
	private AbstractClassifier abstractClassifier;

	public LeakingBaselearnerWrapper(EventBus eventBus, AbstractClassifier abstractClassifier) {
		this.eventBus = eventBus;
		this.abstractClassifier = abstractClassifier;
	}

	@Override
	public void buildClassifier(final Instances instances) throws Exception {
		publishBuildClassifierEvent();
		long startTime = System.currentTimeMillis();

		abstractClassifier.buildClassifier(instances);

		long duration = System.currentTimeMillis() - startTime;
		publishBuildClassifierEvent(duration);
	}

	@Override
	public double classifyInstance(final Instance instance) throws Exception {
		publishClassifyEvent();
		long startTime = System.currentTimeMillis();

		double prediction = abstractClassifier.classifyInstance(instance);

		long duration = System.currentTimeMillis() - startTime;
		publishClassifyEvent(duration);

		return prediction;
	}

	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {
		publishDistributionEvent();
		long startTime = System.currentTimeMillis();

		double[] predictions = abstractClassifier.distributionForInstance(instance);

		long duration = System.currentTimeMillis() - startTime;
		publishDistributionEvent(duration);

		return predictions;
	}

	@Override
	public double[][] distributionsForInstances(Instances batch) throws Exception {
		publishDistributionSEvent();
		long startTime = System.currentTimeMillis();

		double[][] predictions = abstractClassifier.distributionsForInstances(batch);

		long duration = System.currentTimeMillis() - startTime;
		publishDistributionSEvent(duration);

		return predictions;
	}

	@Override
	public Capabilities getCapabilities() {
		return abstractClassifier.getCapabilities();
	}

	@Override
	public void preExecution() throws Exception {
		abstractClassifier.preExecution();
	}

	@Override
	public void postExecution() throws Exception {
		abstractClassifier.postExecution();
	}

	@Override
	public String toSource(String className) throws Exception {
		if (abstractClassifier instanceof Sourcable) {
			return ((Sourcable) abstractClassifier).toString();
		}
		throw new RuntimeException("Abstract classifier " + abstractClassifier.toString() + " does not support sourcing.");
	}

	public void publishClassifyEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.CLASSIFY));
	}

	public void publishClassifyEvent(long duration) {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.CLASSIFY, duration));
	}

	public void publishDistributionEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.DISTRIBUTION));
	}

	public void publishDistributionEvent(long duration) {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.DISTRIBUTION, duration));
	}

	public void publishDistributionSEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.DISTRIBUTIONS));
	}

	public void publishDistributionSEvent(long duration) {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.DISTRIBUTIONS, duration));
	}

	public void publishBuildClassifierEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.BUILD_CLASSIFIER));
	}

	public void publishBuildClassifierEvent(long duration) {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.BUILD_CLASSIFIER, duration));
	}

}
