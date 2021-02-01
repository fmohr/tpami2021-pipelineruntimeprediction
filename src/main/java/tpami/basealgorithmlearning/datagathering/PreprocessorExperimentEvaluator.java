package tpami.basealgorithmlearning.datagathering;

import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.algorithm.Timeout;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.basealgorithmlearning.datagathering.preprocessing.PreprocessorConfigContainer;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.ParametrizedPreprocessorConfigContainer;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;

public class PreprocessorExperimentEvaluator extends AMLAlgorithmExperimentEvaluator {

	/* prepare pre-processor */
	private AttributeSelection as;
	private final String searcherClassName;
	private final String evaluatorClassName;

	public PreprocessorExperimentEvaluator(final PreprocessorConfigContainer container, final Timeout to) {
		super(container, to);
		this.searcherClassName = container.getSearcher();
		this.evaluatorClassName = container.getEvaluator();
	}

	public PreprocessorExperimentEvaluator(final ParametrizedPreprocessorConfigContainer container, final Timeout to) {
		super(container, to);
		this.searcherClassName = container.getSearcher();
		this.evaluatorClassName = container.getEvaluator();
	}

	@Override
	public void fit(final ILabeledDataset<?> trainData, final String optionString, final IExperimentIntermediateResultProcessor processor) throws TrainingException, InterruptedException {
		try {
			String[] parts = optionString.split(";");
			this.as = new AttributeSelection();
			if ((this.searcherClassName.contains("Ranker") || this.searcherClassName.contains("Greedy")) && !parts[0].contains("-N")) {
				parts[0] += " -N 10";
				System.out.println("Setting number of attributes to 10 by default");
			}
			this.as.setSearch(ASSearch.forName(this.searcherClassName, parts[0].trim().split(" ")));
			this.as.setEvaluator(ASEvaluation.forName(this.evaluatorClassName, parts.length > 1 ? parts[1].split(" ") : null));

		} catch (Exception e) {
			throw new IllegalArgumentException(e);
		}
		try {
			Instances inst = ((WekaInstances)trainData).getInstances();
			this.as.SelectAttributes(inst);
			IWekaInstances reducedInstances = new WekaInstances(this.as.reduceDimensionality(inst));
			this.computeAndProcessMetaFeatures(reducedInstances, "_after", processor);
		} catch (Exception e) {
			throw new TrainingException("Fitting the pre-processor failed.", e);
		}
	}

	@Override
	public DescriptiveStatistics apply(final ILabeledDataset<?> applicationData, final int goalTestPoints, final IExperimentIntermediateResultProcessor processor, final SimpleDateFormat format) throws ExperimentEvaluationFailedException {
		DescriptiveStatistics runtimeStats = new DescriptiveStatistics();
		try {
			long start = System.currentTimeMillis();
			this.as.reduceDimensionality(((IWekaInstances)applicationData).getInstances());
			long totalRuntime = System.currentTimeMillis() - start;
			int totalInstances = applicationData.size();
			double runtimePerInstance = totalRuntime * 1.0 / totalInstances;
			for (int i = 0; i < totalInstances; i++) {
				runtimeStats.addValue(runtimePerInstance);
			}
		} catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}
		return runtimeStats;
	}

	@Override
	public void checkFail(final Collection<ExperimentDBEntry> failedExperimentsOnThisDataset, final Map<String, String> experimentKeys) throws ExperimentFailurePredictionException {

		/* what is this experiment about? */
		int datapoints = Integer.parseInt(experimentKeys.get("datapoints"));
		int attributes = Integer.parseInt(experimentKeys.get("attributes"));

		/* currently no fail check implemented */
		this.logger.info("Analyzing {} exceptions of previous experiments.", failedExperimentsOnThisDataset.size());
		AtomicReference<String> reasonString = new AtomicReference<>();
		boolean willFail = failedExperimentsOnThisDataset.stream().anyMatch(e -> {
			int idOfOther = e.getId();
			int numInstancesRequiredByOther = Integer.parseInt(e.getExperiment().getValuesOfKeyFields().get("datapoints"));
			int numAttributesRequiredByOther = Integer.parseInt(e.getExperiment().getValuesOfKeyFields().get("attributes"));
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

	@Override
	public String getNameOfEvaluatedAlgorithm() {
		return this.searcherClassName + "_" + this.evaluatorClassName;
	}

	@Override
	public String getBeforeMFSuffix() {
		return "_before";
	}
}
