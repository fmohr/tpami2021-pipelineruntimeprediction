package tpami.basealgorithmlearning.datagathering;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.timing.TimedComputation;
import tpami.basealgorithmlearning.IConfigContainer;
import tpami.basealgorithmlearning.regression.BasicDatasetFeatureGenerator;
import tpami.basealgorithmlearning.regression.DatasetVarianceFeatureGenerator;

public abstract class AMLAlgorithmExperimentEvaluator implements IExperimentSetEvaluator {

	private static final String DATE_FORMAT = "YYYY-MM-dd HH:mm:ss";
	private final Timeout to;
	private static final int GOAL_TESTPOINTS = 1500;

	/* meta feature generators */
	private static final BasicDatasetFeatureGenerator MGENERATOR_BASIC = new BasicDatasetFeatureGenerator();
	private static final DatasetVarianceFeatureGenerator MGENERATOR_VARIANCE = new DatasetVarianceFeatureGenerator();

	protected Logger logger = LoggerFactory.getLogger(AMLAlgorithmExperimentEvaluator.class);

	private final PeakMemoryObserver mobs = new PeakMemoryObserver();
	private final IConfigContainer container;
	private final ExperimentUtil util = new ExperimentUtil();

	public AMLAlgorithmExperimentEvaluator(final IConfigContainer container, final Timeout to) {
		super();
		this.container = container;
		this.to = to;
		this.mobs.start();
	}

	/* caching of failed experiments */
	private Map<Integer, Collection<ExperimentDBEntry>> knownFailedExperimentsOfDatasets = new HashMap<>();
	private Map<Integer, Long> timestampOfLastErrorQueriesPerDataset = new HashMap<>();

	public abstract String getNameOfEvaluatedAlgorithm();

	// public abstract IClassifier getClassifier() throws Exception;

	public abstract void fit(ILabeledDataset<?> trainData, String[] options, IExperimentIntermediateResultProcessor processor) throws TrainingException, InterruptedException;

	public abstract DescriptiveStatistics apply(ILabeledDataset<?> applicationData, int goalTestPoints, IExperimentIntermediateResultProcessor processor, SimpleDateFormat format) throws ExperimentEvaluationFailedException;

	public abstract String getBeforeMFSuffix();

	protected void computeAndProcessMetaFeatures(final ILabeledDataset<?> data, final String suffix, final IExperimentIntermediateResultProcessor processor) throws Exception {
		Map<String, Object> metaFeatures = new HashMap<>();
		MGENERATOR_BASIC.setSuffix(suffix != null ? suffix : "");
		MGENERATOR_VARIANCE.setSuffix(suffix != null ? suffix : "");
		metaFeatures.putAll(MGENERATOR_BASIC.getFeatureRepresentation(data));
		metaFeatures.putAll(MGENERATOR_VARIANCE.getFeatureRepresentation(data));
		List<String> relevantFeatures = Arrays.asList("totalvariance", "numberofcategories", "numericattributesafterbinarization", "numattributes", "numsymbolicattributes", "attributestocover90pctvariance", "numinstances",
				"attributestocover99pctvariance", "attributestocover50pctvariance", "numlabels", "numnumericattributes", "attributestocover95pctvariance");
		for (String key : new ArrayList<>(metaFeatures.keySet())) {
			if (!relevantFeatures.contains(key)) {
				metaFeatures.remove(key);
			}
		}
		processor.processResults(metaFeatures);
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {

		this.logger.info("Reading in experiment.");
		Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
		int seed = Integer.parseInt(keys.get("seed"));
		int openmlid = Integer.parseInt(keys.get("openmlid"));
		int datapoints = Integer.parseInt(keys.get("datapoints"));
		int attributes = Integer.parseInt(keys.get("attributes"));
		String[] options = keys.get("algorithmoptions").split(" ");

		/* load dataset and create classifier */
		this.logger.info("Running {} on dataset {} with seed {} and {} data points and {} attributes.", this.getNameOfEvaluatedAlgorithm(), openmlid, seed, datapoints, attributes);
		List<ILabeledDataset<?>> splitTmp = null;
		try {

			/* check whether we have reports that even smaller sizes do not work */
			Map<String, Object> comparisonExperiments = new HashMap<>();
			comparisonExperiments.put("openmlid", openmlid);
			if (this.knownFailedExperimentsOfDatasets.containsKey(openmlid)) {
				this.logger.info("Found a cache of failed experiments for this dataset. Checking whether this is already enough to kill the experiment.");
				this.checkFail(this.knownFailedExperimentsOfDatasets.get(openmlid), keys);
			}
			this.logger.info("Did not find any reason to believe that this experiment will fail based on earlier insights.");
			int minTimeInMinutesToWaitBeforeQueryingFailedExperiments = 15;
			if (System.currentTimeMillis() - this.timestampOfLastErrorQueriesPerDataset.computeIfAbsent(openmlid, id -> (long) 0) > 1000 * 60 * minTimeInMinutesToWaitBeforeQueryingFailedExperiments) {
				this.logger.info("Last check was at least {} minutes ago, checking again.", minTimeInMinutesToWaitBeforeQueryingFailedExperiments);
				Collection<ExperimentDBEntry> failedExperimentsOnThisDataset = this.container.getDatabaseHandle().getFailedExperiments(comparisonExperiments);
				this.knownFailedExperimentsOfDatasets.put(openmlid, failedExperimentsOnThisDataset);
				this.timestampOfLastErrorQueriesPerDataset.put(openmlid, System.currentTimeMillis());
				this.checkFail(failedExperimentsOnThisDataset, keys);
				this.logger.info("Did not find any reason to believe that this experiment will fail. Running {} with options {} on dataset {} with seed {} and {} data points.", this.getNameOfEvaluatedAlgorithm(),
						keys.get("algorithmoptions"), openmlid, seed, datapoints);
			} else {
				this.logger.info("Last check was within last {} minutes ago, not checking but conducting (blindly trusting that this experiment is not dominated by some other finished meanwhile).",
						minTimeInMinutesToWaitBeforeQueryingFailedExperiments);
			}

			/* load dataset */
			Dataset ds = (Dataset) OpenMLDatasetReader.deserializeDataset(openmlid);

			/* check whether the experiment is feasible */
			int dataMatrixSize = ds.getNumAttributes() * (datapoints + GOAL_TESTPOINTS);
			int numEntriesOriginal = ds.size() * ds.getNumAttributes();
			this.logger.info("Original dataset has format {}x{} = {} entries. Experiment data matrix will have ({}+{})x{} = {} entries.", ds.size(), ds.getNumAttributes(), numEntriesOriginal, datapoints, GOAL_TESTPOINTS,
					ds.getNumAttributes(), dataMatrixSize);

			/* convert dataset to classification if necessary */
			if (ds.getLabelAttribute() instanceof INumericAttribute) {
				this.logger.info("Converting numeric dataset to classification dataset!");
				ds = (Dataset) DatasetUtil.convertToClassificationDataset(ds);
			}

			/* check whether the dataset is reproducible */
			if (ds.getConstructionPlan().getInstructions().isEmpty()) {
				throw new IllegalStateException("Construction plan for dataset is empty!");
			}
			int goalTestPoints = GOAL_TESTPOINTS;
			if (dataMatrixSize > 5 * Math.pow(10, 8) && numEntriesOriginal < dataMatrixSize) {
				this.logger.warn("Would have to blow-up the dataset to a size of {}, which exceeds the 5*10^8 and hence will not be permitted. Trying to find a smaller test point size.", dataMatrixSize);
				int numDataPointsThatAreAvavilableForTraining = ExperimentUtil.getMaximumInstancesManagableForDataset(ds) - datapoints;
				if (numDataPointsThatAreAvavilableForTraining > 100) {
					goalTestPoints = numDataPointsThatAreAvavilableForTraining;
					this.logger.warn("Adjusting number of test points to {}", numDataPointsThatAreAvavilableForTraining);
				} else if (ds.size() - datapoints >= 100) {
					this.logger.info("We have enough original datapoints to provide {} test cases. We will fall back to these.", ds.size() - datapoints);
					goalTestPoints = ds.size() - datapoints;
				} else {
					this.logger.error("Cannot fit the required datapoints here! Best size for goal test points is {}, which is too few.", numDataPointsThatAreAvavilableForTraining);
					throw new IllegalStateException("Experiment dataset would become too large.");
				}
			}
			splitTmp = this.util.createSizeAdjustedTrainTestSplit(ds, openmlid, datapoints, goalTestPoints, attributes, new Random(seed));

			/* analyze and write meta-features of dataset */
			this.logger.info("Dataset dimension are {}x{} for training and {}x{} for testing", splitTmp.get(0).size(), splitTmp.get(0).getNumAttributes(), splitTmp.get(1).size(), splitTmp.get(1).getNumAttributes());
			this.computeAndProcessMetaFeatures(splitTmp.get(0), this.getBeforeMFSuffix(), processor);

		} catch (Throwable e) {
			throw new ExperimentEvaluationFailedException(e);
		}
		final List<ILabeledDataset<?>> split = splitTmp;

		/* now train classifier */
		SimpleDateFormat format = new SimpleDateFormat(DATE_FORMAT);
		this.logger.info("Start learning.");
		Map<String, Object> map = new HashMap<>();

		long tsTrainStart = System.currentTimeMillis();
		map.put("train_start", format.format(new Date(tsTrainStart)));
		try {
			this.mobs.reset();
			TimedComputation.compute(() -> {
				this.fit(split.get(0), options, processor);
				return null;
			}, new Timeout(this.to.milliseconds(), TimeUnit.MILLISECONDS), "Experiment timeout exceeded.");
			this.logger.info("Stopped learning.");
		} catch (Throwable e) {
			map.put("train_end", format.format(new Date(System.currentTimeMillis())));
			processor.processResults(map);
			throw new ExperimentEvaluationFailedException(e);
		}
		long tsTrainEnd = System.currentTimeMillis();
		map.put("traintimeinms", tsTrainEnd - tsTrainStart);
		map.put("train_end", format.format(new Date(tsTrainEnd)));
		Thread.sleep(2000);
		map.put("memory_peak", this.mobs.getMaxMemoryConsumptionObserved());
		this.logger.info("Finished training, now testing on {} instances. Memory peak was {}", split.get(1).size(), map.get("memory_peak"));
		map.put("test_start", format.format(new Date(System.currentTimeMillis())));
		long timestampStartTesting = System.currentTimeMillis();
		try {
			DescriptiveStatistics runtimeStats = TimedComputation.compute(() -> this.apply(split.get(1), GOAL_TESTPOINTS, processor, format), new Timeout(10, TimeUnit.MINUTES), "Application timed out!");
			long testEnd = System.currentTimeMillis();

			/* check lineality */
			double q3 = runtimeStats.getPercentile(75);
			double q1 = runtimeStats.getPercentile(25);
			double median = runtimeStats.getPercentile(50);
			int iqr = (int) (q3 - q1);
			map.put("predictiontimeiqr", iqr);
			if (q1 > 0 && iqr > median * 0.1) {
				this.logger.warn("IQR GAP: {} - {} = {}", q3, q1, iqr);
			}
			int count1 = 0;
			double tol1 = Math.max(1, Math.ceil(median * 0.01));
			int count2 = 0;
			double tol2 = Math.max(1, Math.ceil(median * 0.05));
			int count3 = 0;
			double tol3 = Math.max(1, Math.ceil(median * 0.10));
			long n = runtimeStats.getN();
			for (double v : runtimeStats.getValues()) {
				if (Math.abs(v - median) <= tol1) {
					count1++;
				}
				if (Math.abs(v - median) <= tol2) {
					count2++;
				}
				if (Math.abs(v - median) <= tol3) {
					count3++;
				}
			}
			if (count1 * 100.0 / n < 50) {
				this.logger.error("Insufficient linearity. Not at least 50% of the predictions are at most {}ms close to the median.", tol1);
			}
			if (count2 * 100.0 / n < 80) {
				this.logger.error("Insufficient linearity. Not at least 80% of the predictions are at most {}ms close to the median.", tol2);
			}
			if (count3 * 100.0 / n < 90) {
				this.logger.error("Insufficient linearity. Not at least 90% of the predictions are at most {}ms close to the median.", tol3);
			}

			/* draw 100 samples of 1000 prediction times. This is a bootstrapping-like check to test the linearity of the predictions */
			if (runtimeStats.getN() >= 1000) {
				DescriptiveStatistics predictionTimeSamplesFor1000 = new DescriptiveStatistics();
				DescriptiveStatistics predictionTimeSamplesFor100 = new DescriptiveStatistics();
				Collection<Double> values = new ArrayList<>();
				for (double d : runtimeStats.getValues()) {
					values.add(d);
				}
				for (int i = 0; i < 100; i++) {
					predictionTimeSamplesFor1000.addValue(SetUtil.getRandomSubset(values, 1000, new Random(i)).stream().reduce((a, b) -> a + b).get());
					predictionTimeSamplesFor100.addValue(SetUtil.getRandomSubset(values, 100, new Random(i)).stream().reduce((a, b) -> a + b).get());
				}
				map.put("predictiontimeinmsperkinstances", (int) (predictionTimeSamplesFor1000.getMean()));
				map.put("stdon100predictionsinms", (int) predictionTimeSamplesFor100.getStandardDeviation());
				map.put("stdonkpredictionsinms", (int) predictionTimeSamplesFor1000.getStandardDeviation());
			} else {
				map.put("predictiontimeinmsperkinstances", (int) (runtimeStats.getMean() * 10000));
				map.put("stdon100predictionsinms", -1);
				map.put("stdonkpredictionsinms", -1);
			}

			map.put("test_end", format.format(new Date(testEnd)));
			map.put("predictedinstances", n);
			map.put("timeforpredictionsinms", testEnd - timestampStartTesting);
			map.put("linearityconfidence1", (int) (count1 * 100.0 / n));
			map.put("linearityconfidence2", (int) (count2 * 100.0 / n));
			map.put("linearityconfidence3", (int) (count3 * 100.0 / n));
			map.put("medianpredictiontimeinms", (int) median);
			map.put("stdinpredictiontimeminms", (int) runtimeStats.getStandardDeviation());
			processor.processResults(map);
			this.logger.info("Finished Experiment {}. Results: {}", experimentEntry.getExperiment().getValuesOfKeyFields(), map);
		} catch (AlgorithmTimeoutedException | ExecutionException e) {
			map.put("test_end", format.format(new Date(System.currentTimeMillis())));
			processor.processResults(map);
			throw new ExperimentEvaluationFailedException(e);
		}
	}

	/* We can omit this execution in any of the following cases:
	 * - an earlier execution with less datapoints on the same dataset was canceled because there are not enough datapoints
	 * - an earlier execution with less datapoints on the same dataset AND the same algorithm options has timed out
	 **/
	public abstract void checkFail(final Collection<ExperimentDBEntry> failedExperimentsOnThisDataset, final Map<String, String> experimentKeys) throws ExperimentFailurePredictionException;

	public IConfigContainer getContainer() {
		return this.container;
	}
}
