package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.Map;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.basealgorithmlearning.regression.BasicDatasetFeatureGenerator;
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
		publishStartComputeMetafeaturesEvent();

		Map<String, Object> metafeatures = computeMetafeatures(instances);

		publishStopComputeMetafeaturesEvent(metafeatures);

		publishStartBuildClassifierEvent();

		abstractClassifier.buildClassifier(instances);

		publishStopBuildClassifierEvent();
	}

	@Override
	public double classifyInstance(final Instance instance) throws Exception {
		publishStartClassifyEvent();

		double prediction = abstractClassifier.classifyInstance(instance);

		publishStopClassifyEvent();

		return prediction;
	}

	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {
		publishStartDistributionEvent();

		double[] predictions = abstractClassifier.distributionForInstance(instance);

		publishStopDistributionEvent();

		return predictions;
	}

	@Override
	public double[][] distributionsForInstances(Instances batch) throws Exception {
		publishStartDistributionSEvent();

		double[][] predictions = abstractClassifier.distributionsForInstances(batch);

		publishStopDistributionSEvent();

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

	public void publishStartClassifyEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_CLASSIFY, this));
	}

	public void publishStopClassifyEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_CLASSIFY, this));
	}

	public void publishStartDistributionEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_DISTRIBUTION, this));
	}

	public void publishStopDistributionEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_DISTRIBUTION, this));
	}

	public void publishStartDistributionSEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_DISTRIBUTIONS, this));
	}

	public void publishStopDistributionSEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_DISTRIBUTIONS, this));
	}

	public void publishStartBuildClassifierEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_BUILD_CLASSIFIER, this));
	}

	public void publishStopBuildClassifierEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_BUILD_CLASSIFIER, this));
	}

	public void publishStartComputeMetafeaturesEvent() {
		eventBus.post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_METAFEATURE_COMPUTATION, this));
	}

	public void publishStopComputeMetafeaturesEvent(Map<String, Object> metafeatures) {
		eventBus.post(new LeakingBaselearnerEvent(metafeatures, this));
	}

	public Map<String, Object> computeMetafeatures(Instances instances) {
		BasicDatasetFeatureGenerator featureGenerator = new BasicDatasetFeatureGenerator();
		return featureGenerator.getFeatureRepresentation(new WekaInstances(instances));
	}

}
