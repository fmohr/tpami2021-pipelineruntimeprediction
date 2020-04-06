package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.concurrent.TimeUnit;

import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;

public class DefaultMetaLearnerExperimenter {

	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout TIME_OUT = new Timeout(1, TimeUnit.HOURS);

	public static void main(final String[] args) throws Exception {

		// /* prepare database */
		// ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(config, databaseHandle);
		// preparer.setLoggerName("example");
		// preparer.synchronizeExperiments();
		// System.exit(0);

		DefaultMetaLearnerConfigContainer container = new DefaultMetaLearnerConfigContainer(args[0], args[1]);
		Class<?> classifierClass = Class.forName(args[1]);
		IExperimentDatabaseHandle databaseHandle = container.getDatabaseHandle();

		/* run an experiment */

		LOGGER.info("Creating the runner.");
		ExperimentRunner runner = new ExperimentRunner(container.getConfig(), new DefaultMetaLearnerExperimentSetEvaluator(container, classifierClass, TIME_OUT), databaseHandle);

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
