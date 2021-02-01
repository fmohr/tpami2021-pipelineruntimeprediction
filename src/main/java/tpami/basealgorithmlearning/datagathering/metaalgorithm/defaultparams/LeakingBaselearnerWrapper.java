package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.Map;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.basealgorithmlearning.datagathering.BasicDatasetFeatureGenerator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.WeightedInstancesHandler;

public class LeakingBaselearnerWrapper extends AbstractClassifier implements Sourcable, Randomizable, WeightedInstancesHandler {

	private int seed;

	private String randomString;

	private transient EventBus eventBus;
	private Classifier abstractClassifier;


	public LeakingBaselearnerWrapper(final EventBus eventBus, final Classifier abstractClassifier, final String randomString) {
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
			this.publishStartComputeMetafeaturesEvent();

			Map<String, Object> metafeatures = this.computeMetafeatures(instances);

			this.publishStopComputeMetafeaturesEvent(metafeatures);

			this.publishStartBuildClassifierEvent();

			this.abstractClassifier.buildClassifier(instances);

			this.publishStopBuildClassifierEvent();

		} catch (Exception ex) {
			this.publishExceptionEvent(ex);
			throw ex;
		}
	}

	@Override
	public double classifyInstance(final Instance instance) throws Exception {
		try {
			this.publishStartClassifyEvent();

			double prediction = this.abstractClassifier.classifyInstance(instance);

			this.publishStopClassifyEvent();

			return prediction;
		} catch (Exception ex) {
			this.publishExceptionEvent(ex);
			throw ex;
		}

	}

	@Override
	public double[] distributionForInstance(final Instance instance) throws Exception {
		try {
			this.publishStartDistributionEvent();

			double[] predictions = this.abstractClassifier.distributionForInstance(instance);

			this.publishStopDistributionEvent();

			return predictions;
		} catch (Exception ex) {
			this.publishExceptionEvent(ex);
			throw ex;
		}
	}

	@Override
	public double[][] distributionsForInstances(final Instances batch) throws Exception {
		try {
			if (this.abstractClassifier instanceof AbstractClassifier) {
				this.publishStartDistributionSEvent();

				double[][] predictions = ((AbstractClassifier) this.abstractClassifier).distributionsForInstances(batch);

				this.publishStopDistributionSEvent();

				return predictions;
			}

			throw new RuntimeException("Classifier " + this.abstractClassifier + " does not support distributionS.");
		} catch (Exception ex) {
			this.publishExceptionEvent(ex);
			throw ex;
		}
	}

	@Override
	public Capabilities getCapabilities() {
		return this.abstractClassifier.getCapabilities();
	}

	@Override
	public String toSource(final String className) throws Exception {
		try {
			if (this.abstractClassifier instanceof Sourcable) {
				return ((Sourcable) this.abstractClassifier).toString();
			}
			throw new RuntimeException("Abstract classifier " + this.abstractClassifier.toString() + " does not support sourcing.");
		} catch (Exception ex) {
			this.publishExceptionEvent(ex);
			throw ex;
		}
	}

	@Override
	public void setSeed(final int seed) {
		if (this.abstractClassifier instanceof Randomizable) {
			((Randomizable) this.abstractClassifier).setSeed(seed);
		} else {
			this.seed = seed;
		}

	}

	@Override
	public int getSeed() {
		if (this.abstractClassifier instanceof Randomizable) {
			return ((Randomizable) this.abstractClassifier).getSeed();
		}
		return this.seed;
	}

	public void publishStartClassifyEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_CLASSIFY, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStopClassifyEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_CLASSIFY, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStartDistributionEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_DISTRIBUTION, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStopDistributionEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_DISTRIBUTION, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStartDistributionSEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_DISTRIBUTIONS, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStopDistributionSEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_DISTRIBUTIONS, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStartBuildClassifierEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_BUILD_CLASSIFIER, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStopBuildClassifierEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.STOP_BUILD_CLASSIFIER, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStartComputeMetafeaturesEvent() {
		this.getEventBus().post(new LeakingBaselearnerEvent(ELeakingBaselearnerEventType.START_METAFEATURE_COMPUTATION, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishStopComputeMetafeaturesEvent(final Map<String, Object> metafeatures) {
		this.getEventBus().post(new LeakingBaselearnerEvent(metafeatures, this, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public void publishExceptionEvent(final Exception exception) {
		this.getEventBus().post(new LeakingBaselearnerEvent(exception, EventBusHolder.isMetaLearnerTrained(this.randomString)));
	}

	public Map<String, Object> computeMetafeatures(final Instances instances) {
		BasicDatasetFeatureGenerator featureGenerator = new BasicDatasetFeatureGenerator();
		return featureGenerator.getFeatureRepresentation(new WekaInstances(instances));
	}

	private EventBus getEventBus() {
		if (this.eventBus == null) {
			this.eventBus = EventBusHolder.getEventBusForWrapper(this);
		}
		return this.eventBus;
	}

	public String getRandomString() {
		return this.randomString;
	}

	public void informThatMetaLearnerHasCompletedTraining() {
		EventBusHolder.publishFactThatMetaLearnerHasFinishedTraining(this.randomString);
	}

}
