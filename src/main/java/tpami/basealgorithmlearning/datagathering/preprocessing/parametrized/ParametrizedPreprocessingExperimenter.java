package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized;
import java.util.concurrent.TimeUnit;

import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.logging.LoggerUtil;
import tpami.basealgorithmlearning.datagathering.PreprocessorExperimentEvaluator;

public class ParametrizedPreprocessingExperimenter {

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
		ParametrizedPreprocessorConfigContainer container = new ParametrizedPreprocessorConfigContainer(configFile, searcherClassName, evaluatorClassName);
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
}
