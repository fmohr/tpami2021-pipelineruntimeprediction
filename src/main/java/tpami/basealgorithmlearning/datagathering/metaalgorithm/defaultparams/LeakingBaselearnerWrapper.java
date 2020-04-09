package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.Map;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.basealgorithmlearning.regression.BasicDatasetFeatureGenerator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LeakingBaselearnerWrapper extends AbstractClassifier implements Sourcable {

	private String randomString;

	private transient EventBus eventBus;
	private Classifier abstractClassifier;

	public LeakingBaselearnerWrapper(EventBus eventBus, Classifier abstractClassifier, String randomString) {
		this.eventBus = eventBus;
		this.abstractClassifier = abstractClassifier;
		this.randomString = randomString;
		if (eventBus != null) {
			EventBusHolder.registerEventBus(randomString, eventBus);
		}
	}

	@Override
	public void buildClassifier(final Instances instances) throws Exception {
		try {
			publishStartComputeMetafeaturesEvent();

			Map<String, Object> metafeatures = computeMetafeatures(instances);

			publishStopComputeMetafeaturesEvent(metafeatures);

			publishStartBuildClassifierEvent();

			abstractClassifier.buildClassifier(instances);

			publishStopBuildClassifierEvent();

		} catch (Exception ex) {
			publishExceptionEvent(ex);
			throw ex;
		}
	}

	@Override
	public double classifyInstance(final Instance instance) throws Exception {
		try {
			publishStartClassifyEvent();

			double prediction = abstractClassifier.classifyInstance(instance);

			publishStopClassifyEvent();

			return prediction;
		} catch (Exception ex) {
			publishExceptionEvent(ex);
			throw ex;
		}

	}

	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {
		try {
			publishStartDistributionEvent();

			double[] predictions = abstractClassifier.distributionForInstance(instance);

			publishStopDistributionEvent();

			return predictions;
		} catch (Exception ex) {
			publishExceptionEvent(ex);
			throw ex;
		}
	}

	@Override
	public double[][] distributionsForInstances(Instances batch) throws Exception {
		try {
			if (abstractClassifier instanceof AbstractClassifier) {
				publishStartDistributionSEvent();

				double[][] predictions = ((AbstractClassifier) abstractClassifier).distributionsForInstances(batch);

				publishStopDistributionSEvent();

				return predictions;
			}

			throw new RuntimeException("Classifier " + abstractClassifier + " does not support distributionS.");
		} catch (Exception ex) {
			publishExceptionEvent(ex);
			throw ex;
		}
	}

	@Override
	public Capabilities getCapabilities() {
		return abstractClassifier.getCapabilities();
	}

	@Override
	public String toSource(String className) throws Exception {
		try {
			if (abstractClassifier instanceof Sourcable) {
				return ((Sourcable) abstractClassifier).toString();
			}
			throw new RuntimeException("Abstract classifier " + abstractClassifier.toString() + " does not support sourcing.");
		} catch (Exception ex) {
			publishExceptionEvent(ex);
			throw ex;
		}
	}

	public void publishStartClassifyEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_CLASSIFY, this));
	}

	public void publishStopClassifyEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_CLASSIFY, this));
	}

	public void publishStartDistributionEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_DISTRIBUTION, this));
	}

	public void publishStopDistributionEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_DISTRIBUTION, this));
	}

	public void publishStartDistributionSEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_DISTRIBUTIONS, this));
	}

	public void publishStopDistributionSEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_DISTRIBUTIONS, this));
	}

	public void publishStartBuildClassifierEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_BUILD_CLASSIFIER, this));
	}

	public void publishStopBuildClassifierEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_BUILD_CLASSIFIER, this));
	}

	public void publishStartComputeMetafeaturesEvent() {
		getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_METAFEATURE_COMPUTATION, this));
	}

	public void publishStopComputeMetafeaturesEvent(Map<String, Object> metafeatures) {
		getEventBus().post(new LeakingBaselearnerEvent(metafeatures, this));
	}

	public void publishExceptionEvent(Exception exception) {
		getEventBus().post(new LeakingBaselearnerEvent(exception));
	}

	public Map<String, Object> computeMetafeatures(Instances instances) {
		BasicDatasetFeatureGenerator featureGenerator = new BasicDatasetFeatureGenerator();
		return featureGenerator.getFeatureRepresentation(new WekaInstances(instances));
	}

	private EventBus getEventBus() {
		if (eventBus == null) {
			eventBus = EventBusHolder.getEventBusForWrapper(this);
		}
		return eventBus;
	}

	public String getRandomString() {
		return randomString;
	}

}
