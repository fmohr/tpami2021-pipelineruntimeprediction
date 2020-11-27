package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.jaicore.timing.TimedComputation;
import tpami.basealgorithmlearning.datagathering.PeakMemoryObserver;
import tpami.basealgorithmlearning.regression.BasicDatasetFeatureGenerator;
import tpami.basealgorithmlearning.regression.DatasetVarianceFeatureGenerator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.core.Instances;
import weka.core.OptionHandler;

public class ParametrizedPreprocessingExperimenter {

	private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
	private static final Logger LOGGER = LoggerFactory.getLogger("example");
	private static IExperimentDatabaseHandle databaseHandle;
	private static final int TOTAL_RUNTIME_IN_SECONDS = 3600 * 12;
	private static final Timeout to = new Timeout(1, TimeUnit.HOURS);
	private static Class<?> searcherClass;
	private static Class<?> evaluatorClass;

	private static Map<Integer, Collection<ExperimentDBEntry>> knownFailedExperimentsOfDatasets = new HashMap<>();

	public static void main(final String[] args) throws Exception {

		/* get experiment configuration */
		String configFile = args[0];
		searcherClass = Class.forName(args[1]);
		evaluatorClass = Class.forName(args[2]);

		ParametrizedPreprocessorConfigContainer container = new ParametrizedPreprocessorConfigContainer(configFile, searcherClass.getSimpleName(), evaluatorClass.getSimpleName());
		databaseHandle = container.getDatabaseHandle();


		/* starting memory observer */
		LOGGER.info("Creating memory observer");
		final PeakMemoryObserver mobs = new PeakMemoryObserver();
		mobs.start();

		final BasicDatasetFeatureGenerator fmBasic = new BasicDatasetFeatureGenerator();
		final DatasetVarianceFeatureGenerator fmVariance = new DatasetVarianceFeatureGenerator();


		/* run an experiment */
		LOGGER.info("Creating the runner.");

		ExperimentRunner runner = new ExperimentRunner(container.getConfig(), new IExperimentSetEvaluator() {

			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException, ExperimentFailurePredictionException {
				try {
					LOGGER.info("Reading in experiment.");
					Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
					int seed = Integer.parseInt(keys.get("seed"));
					int openmlid = Integer.parseInt(keys.get("openmlid"));
					int datapoints = Integer.parseInt(keys.get("datapoints"));
					String evaloptions= keys.get("evaloptions");
					System.out.println(evaloptions);

					/* first check whether the experiment on this dataset can possibly be successful */
					//					Map<String, Object> comparisonExperiments = new HashMap<>();
					//					comparisonExperiments.put("openmlid", openmlid);
					//					LOGGER.info("Getting and analyzing failed experiments on this dataset.");
					//					checkFail(knownFailedExperimentsOfDatasets.get(openmlid), datapoints, attributes);
					//					Collection<ExperimentDBEntry> failedExperimentsOnThisDataset = databaseHandle.getFailedExperiments(comparisonExperiments);
					//					knownFailedExperimentsOfDatasets.put(openmlid, failedExperimentsOnThisDataset);
					//					checkFail(failedExperimentsOnThisDataset, datapoints, attributes);

					//					LOGGER.info("Did not find any reason to believe that this experiment will fail. Running {}/{} on dataset {} with seed {}, {} data points and {} attributes.", searcherClass.getSimpleName(), evaluatorClass.getSimpleName(), openmlid, seed, datapoints, attributes);
					Map<String, Object> map = new HashMap<>();

					/* load dataset and reduce dimensionality */
					ILabeledDataset<?> ds = OpenMLDatasetReader.deserializeDataset(openmlid);
					//					if (ds.getNumAttributes() < attributes) {
					//						throw new IllegalExperimentSetupException("Dataset has only " + ds.getNumAttributes() + " attributes, so cannot conduct experiment with " + attributes + " attributes.");
					//					}
					if (ds.size() < datapoints) {
						throw new IllegalExperimentSetupException("Dataset has only " + ds.size() + " points, so cannot conduct experiment with " + datapoints + " data points.");
					}

					/* reducing number of columns and number of rows */
					Random r = new Random(seed);
					//					while (ds.getNumAttributes() > attributes) {
					//						ds.removeColumn(r.nextInt(ds.getNumAttributes()));
					//					}

					ILabeledDataset<?> relevantFold = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, datapoints * 1.0 / ds.size()).get(0);
					LOGGER.info("Having relevant fold of size {} and with {} attributes.", relevantFold.size(), relevantFold.getNumAttributes());
					final IWekaInstances data = new WekaInstances(relevantFold);
					//					if (data.getNumAttributes() > attributes) {
					//						throw new IllegalStateException("Attribute reduction was not maintained by Weka split!");
					//					}

					/* prepare pre-processor */
					final AttributeSelection as = new AttributeSelection();
					String[] searchOptions = null;
					if (evaluatorClass == CfsSubsetEval.class) {
						if (searcherClass == BestFirst.class) {
							String strOptions = "";
							strOptions += "-D " + keys.get("bf_d");
							strOptions += " -N " + keys.get("bf_n");
							strOptions += " -S " + keys.get("bf_s");
							searchOptions = strOptions.split(" ");
						}
						if (searcherClass == GreedyStepwise.class) {
							String strOptions = "";
							if (keys.get("gsw_c").equals("1")) {
								strOptions += "-C";
							}
							if (keys.get("gsw_b").equals("1")) {
								if (strOptions.length() > 0) {
									strOptions += " ";
								}
								strOptions += "-B";
							}
							if (strOptions.length() > 0) {
								strOptions += " ";
							}
							strOptions += "-N " + keys.get("gsw_n");
							searchOptions = strOptions.split(" ");
						}
					}
					else {
						searchOptions = ("-N " + keys.get("rankern")).split(" ");
					}
					if (searchOptions == null) {
						System.exit(1);
					}
					ASSearch search = ASSearch.forName(searcherClass.getName(), searchOptions);
					ASEvaluation evaluator = ASEvaluation.forName(evaluatorClass.getName(), evaloptions != null ? evaloptions.split(" ") : null);
					as.setSearch(search);
					as.setEvaluator(evaluator);

					/* register meta-features before */
					fmBasic.setSuffix("_before");
					map.putAll(fmBasic.getFeatureRepresentation(data));
					fmVariance.setSuffix("_before");
					map.putAll(fmVariance.getFeatureRepresentation(data));
					processor.processResults(map);
					LOGGER.info("Updated meta features of dataset: {}", map);
					map.clear();

					/* now run pre-processor */
					mobs.reset();
					LOGGER.info("Starting computation");
					if (Math.abs(data.getInstances().size() - datapoints) > 1) {
						throw new IllegalStateException("Data points are " + data.getInstances().size() + " but should be " + datapoints);
					}
					LOGGER.info("Searcher: {} with options {}", search, Arrays.toString(((OptionHandler)search).getOptions()));
					LOGGER.info("Evaluator: {} with options {}", evaluator.getClass().getName(), Arrays.toString(((OptionHandler)evaluator).getOptions()));
					map.put("time_alg_start", DATE_FORMAT.format(new Date(System.currentTimeMillis())));
					TimedComputation.compute(() -> { as.SelectAttributes(data.getInstances()); return null;}, to, "Experiment timeout exceeded.");
					map.put("time_alg_end", DATE_FORMAT.format(new Date(System.currentTimeMillis())));
					map.put("time_alg_apply_start", DATE_FORMAT.format(new Date(System.currentTimeMillis())));
					LOGGER.info("Algorithm finished at {}. Now computing features of dataset with reduced dimensionality.", map.get("time_alg_end"));
					Instances inst = as.reduceDimensionality(data.getInstances());
					map.put("time_alg_apply_end", DATE_FORMAT.format(new Date(System.currentTimeMillis())));
					LOGGER.info("Reduction ratio is {}/{}", inst.numAttributes(), data.getInstances().numAttributes());
					IWekaInstances reducedInstances = new WekaInstances(inst);
					fmBasic.setSuffix("_after");
					map.putAll(fmBasic.getFeatureRepresentation(reducedInstances));
					fmVariance.setSuffix("_after");
					map.putAll(fmVariance.getFeatureRepresentation(data));
					processor.processResults(map);
					LOGGER.info("Updated after-meta features of dataset: {}", map);
					map.clear();
					Thread.sleep(1000);

					map.put("memory_peak", mobs.getMaxMemoryConsumptionObserved());
					processor.processResults(map);
					LOGGER.info("Finished Experiment {}. Results: {}", experimentEntry.getExperiment().getValuesOfKeyFields(), map);
				}
				catch (ExperimentFailurePredictionException e) {
					throw e;
				}
				catch (Throwable e) {
					System.err.println("ENCAPSUALTING ERROR!");
					e.printStackTrace();
					throw new ExperimentEvaluationFailedException(e);
				}
			}
		}, databaseHandle);

		LOGGER.info("Running random experiments with timeout " + TOTAL_RUNTIME_IN_SECONDS + "s.");
		runner.setLoggerName("example");
		long start = System.currentTimeMillis();
		int i = 0;
		while ((System.currentTimeMillis() - start) / 1000 <= TOTAL_RUNTIME_IN_SECONDS - to.seconds() * 1.1) {
			LOGGER.info("Conducting next experiment.");
			try {
				runner.randomlyConductExperiments(20);
				if (++i % 1000 == 0 && databaseHandle.getOpenExperiments().isEmpty()) {
					LOGGER.info("No more experiments.");
					System.exit(0);
				}
			}
			catch (Throwable e) {
				e.printStackTrace();
			}
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
