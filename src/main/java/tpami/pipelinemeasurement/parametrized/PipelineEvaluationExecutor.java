package tpami.pipelinemeasurement.parametrized;

import java.util.concurrent.TimeUnit;

import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.logging.LoggerUtil;
import tpami.basealgorithmlearning.IConfigContainer;

public class PipelineEvaluationExecutor {

	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout to = new Timeout(1, TimeUnit.HOURS);
	private static final Logger LOGGER = LoggerFactory.getLogger(LoggerUtil.LOGGER_NAME_EVALUATOR);

	public static void main(final String[] args) throws Exception {

		/* parse arguments */
		String configFileName = args[0];
		String executorinfo = args[1];

		/* create evaluator and runner */
		PipelineExperimentEvaluator evaluator = new PipelineExperimentEvaluator(new PipelineConfigContainer(configFileName), to);
		IConfigContainer container = evaluator.getContainer();
		ExperimentRunner runner = new ExperimentRunner(container.getExperimentSetConfig(), evaluator, container.getDatabaseHandle(), executorinfo);

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
}
