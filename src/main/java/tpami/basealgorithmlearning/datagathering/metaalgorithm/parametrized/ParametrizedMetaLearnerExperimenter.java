package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized;

import java.util.concurrent.TimeUnit;

import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;

public class ParametrizedMetaLearnerExperimenter {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout TIME_OUT = new Timeout(1, TimeUnit.HOURS);

	public static void main(final String[] args) throws Exception {

		ParametrizedMetaLearnerConfigContainer container = new ParametrizedMetaLearnerConfigContainer(args[0], args[1]);
		Class<?> metalearnerClass = Class.forName(args[1]);
		IExperimentDatabaseHandle databaseHandle = container.getDatabaseHandle();
		String executorDetails = args[2];

		LOGGER.info("Creating the runner. Executor details: {}", executorDetails);
		ParametrizedMetaLearnerExperimentSetEvaluator evaluator = new ParametrizedMetaLearnerExperimentSetEvaluator(container, metalearnerClass, TIME_OUT, executorDetails);
		evaluator.setLoggerName("evaluator");
		ExperimentRunner runner = new ExperimentRunner(container.getConfig(), evaluator, databaseHandle);

		LOGGER.info("Runner created.");

		LOGGER.info("Running random experiments with timeout " + TOTAL_RUNTIME_IN_SECONDS + "s.");
		runner.setLoggerName("example");
		long start = System.currentTimeMillis();
		while ((System.currentTimeMillis() - start) / 1000 <= TOTAL_RUNTIME_IN_SECONDS - TIME_OUT.seconds() * 1.1) {
			LOGGER.info("Conducting next experiment.");
			try {
				runner.randomlyConductExperiments(1);
			} catch (Throwable e) {
				e.printStackTrace();
			}
		}
		LOGGER.info("No more time left to conduct more experiments. Stopping.");
		System.exit(0);
	}

}
