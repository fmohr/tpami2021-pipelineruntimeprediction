package tpami.basealgorithmlearning.datagathering;

import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.classification.IClassifier;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.algorithm.Timeout;

import ai.libs.jaicore.basic.MathExt;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import tpami.basealgorithmlearning.IConfigContainer;

public abstract class ALearnerExperimentEvaluator extends AMLAlgorithmExperimentEvaluator {

	private IClassifier classifier;

	public ALearnerExperimentEvaluator(final IConfigContainer container, final Timeout to) {
		super(container, to);
	}

	@Override
	public void fit(final ILabeledDataset<?> trainData, final String optionString, final IExperimentIntermediateResultProcessor processor) throws TrainingException, InterruptedException {
		try {
			this.classifier = this.getClassifier(optionString);
		} catch (Exception e) {
			throw new IllegalArgumentException(e);
		}
		this.classifier.fit(trainData);
	}

	public abstract IClassifier getClassifier(String optionString) throws Exception;

	@Override
	public DescriptiveStatistics apply(final ILabeledDataset<?> applicationData, final int goalTestPoints, final IExperimentIntermediateResultProcessor processor, final SimpleDateFormat format) throws ExperimentEvaluationFailedException {
		DescriptiveStatistics runtimeStats = new DescriptiveStatistics();
		try {
			long timestampStartTesting = System.currentTimeMillis();
			long lastTimeoutCheck = 0;
			int n = applicationData.size();
			for (ILabeledInstance i : applicationData) {
				long now = System.currentTimeMillis();
				this.classifier.predict(i).getPrediction();
				long after = System.currentTimeMillis();
				runtimeStats.addValue(after - (double) now);
				if (now - lastTimeoutCheck > 10000) {
					if (now - timestampStartTesting > 5 * 60 * 1000 || runtimeStats.getN() > goalTestPoints) {
						this.logger.debug("Early finishing run because of time issues.");
						break;
					}
					lastTimeoutCheck = now;
					this.logger.debug("Current Prediction Progress: {}/{} ({}%)", runtimeStats.getN(), n, MathExt.round(runtimeStats.getN() * 100.0 / n, 2));
				}
			}
		} catch (Throwable e) {
			throw new ExperimentEvaluationFailedException(e);
		}
		return runtimeStats;
	}

	@Override
	public void checkFail(final Collection<ExperimentDBEntry> failedExperimentsOnThisDataset, final Map<String, String> experimentKeys) throws ExperimentFailurePredictionException {

		/* what is this experiment about? */
		int datapoints = Integer.parseInt(experimentKeys.get("datapoints"));
		int attributes = Integer.parseInt(experimentKeys.get("attributes"));
		String options = experimentKeys.get("algorithmoptions");

		/* currently no fail check implemented */
		this.logger.info("Analyzing {} exceptions of previous experiments.", failedExperimentsOnThisDataset.size());
		AtomicReference<String> reasonString = new AtomicReference<>();
		boolean willFail = failedExperimentsOnThisDataset.stream().anyMatch(e -> {
			int idOfOther = e.getId();
			int numInstancesRequiredByOther = Integer.parseInt(e.getExperiment().getValuesOfKeyFields().get("datapoints"));
			int numAttributesRequiredByOther = Integer.parseInt(e.getExperiment().getValuesOfKeyFields().get("attributes"));
			String algorithmoptions = e.getExperiment().getValuesOfKeyFields().get("algorithmoptions");
			boolean hasSameParameters = (algorithmoptions == null && options == null) || (algorithmoptions != null && options != null && algorithmoptions.equals(options));
			if (!hasSameParameters) {
				return false;
			}
			boolean requiresAtLeastAsManyPointsThanFailed = numInstancesRequiredByOther <= datapoints;
			boolean requiresAtLeastAsManyAttributesThanFailed = numAttributesRequiredByOther <= attributes;
			String errorMsg = e.getExperiment().getError().toLowerCase();

			/* check on timeout */
			boolean otherTimedOut = errorMsg.contains("timeout");
			if (requiresAtLeastAsManyPointsThanFailed && requiresAtLeastAsManyAttributesThanFailed && otherTimedOut) {
				reasonString.set("This has at least as much instances and attributes as " + idOfOther + ", which failed due to a timeout.");
				return true;
			}

			/* check on memory overflow */
			boolean otherHasOverflow = errorMsg.contains("memory");
			if (requiresAtLeastAsManyPointsThanFailed && requiresAtLeastAsManyAttributesThanFailed && otherHasOverflow) {
				reasonString.set("This has at least as much instances and attributes as " + idOfOther + ", which ran out of memory!");
				return true;
			}
			return false;
		});
		if (willFail) {
			this.logger.warn("Announcing fail with reason: {}", reasonString.get());
			throw new ExperimentFailurePredictionException("Experiment will fail for the following reason: " + reasonString.get());
		}
	}
}
