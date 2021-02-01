package tpami.basealgorithmlearning.datagathering.preprocessing;

import java.util.Collection;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.logging.LoggerUtil;
import tpami.basealgorithmlearning.datagathering.PreprocessorExperimentEvaluator;

public class PreprocessingExperimenter {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout to = new Timeout(1, TimeUnit.HOURS);

	public static void main(final String[] args) throws Exception {

		/* get experiment configuration */
		String configFile = args[0];
		String searcherClassName = args[1];
		String evaluatorClassName = args[2];
		String executorinfo = args[3];

		/* run an experiment */
		PreprocessorConfigContainer container = new PreprocessorConfigContainer(configFile, searcherClassName, evaluatorClassName);
		LOGGER.info("Creating the runner.");
		ExperimentRunner runner = new ExperimentRunner(container.getConfig(), new PreprocessorExperimentEvaluator(container, to), container.getDatabaseHandle(), executorinfo);


		/* run experiments */
		LOGGER.info("Running random experiments with timeout " + TOTAL_RUNTIME_IN_SECONDS + "s.");
		runner.setLoggerName(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);
		long start = System.currentTimeMillis();
		while (runner.mightHaveMoreExperiments() && (System.currentTimeMillis() - start) / 1000 <= TOTAL_RUNTIME_IN_SECONDS - to.seconds() * 1.1) {
			LOGGER.info("Conducting next experiment.");
			runner.randomlyConductExperiments(1);
		}
		LOGGER.info("No more time left to conduct more experiments. Stopping.");
		System.exit(0);
	}

	public static void checkFail(final Collection<ExperimentDBEntry> failedExperimentsOnThisDataset, final int datapoints, final int attributes) throws ExperimentFailurePredictionException {
		if (failedExperimentsOnThisDataset == null) {
			return;
		}
		AtomicReference<String> reasonString = new AtomicReference<>();
		boolean willFail = failedExperimentsOnThisDataset.stream().anyMatch(e -> {
			int idOfOther = e.getId();
			boolean requiresAtLeastAsManyAttributesThanFailed = Integer.parseInt(e.getExperiment().getValuesOfKeyFields().get("attributes")) <= attributes;
			boolean requiresAtLeastAsManyPointsThanFailed = Integer.parseInt(e.getExperiment().getValuesOfKeyFields().get("datapoints")) <= datapoints;
			String errorMsg = e.getExperiment().getError().toLowerCase();
			boolean otherHadTooFewInstances = errorMsg.contains("specified sample size is bigger than");
			if (requiresAtLeastAsManyPointsThanFailed && otherHadTooFewInstances) {
				reasonString.set("This experiment requires at least as many instances as " + idOfOther + ", which failed because it demanded too many instances.");
				return true;
			}
			boolean otherHadTooFewAttributes = errorMsg.contains("attributes, so cannot conduct experiment with");
			if (requiresAtLeastAsManyAttributesThanFailed && otherHadTooFewAttributes) {
				reasonString.set("This experiment requires at least as many attributes as " + idOfOther + ", which failed because it demanded too many attributes.");
				return true;
			}
			boolean otherTimedOut = errorMsg.contains("timeout");
			boolean isAtLeastAsBigAsOther = !requiresAtLeastAsManyAttributesThanFailed && !requiresAtLeastAsManyPointsThanFailed;
			if (isAtLeastAsBigAsOther && otherTimedOut) {
				reasonString.set("This has at least as much attributes and instances as " + idOfOther + ", which failed due to a timeout.");
				return true;
			}
			return otherTimedOut;
		});
		if (willFail) {
			LOGGER.warn("Announcing fail with reason: {}", reasonString.get());
			throw new ExperimentFailurePredictionException("Experiment will fail for the following reason: " + reasonString.get());
		}
	}
}
