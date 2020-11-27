package tpami.pipelinemeasurement;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.MathExt;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.classification.pipeline.MLPipeline;
import ai.libs.jaicore.timing.TimedComputation;
import tpami.basealgorithmlearning.datagathering.PeakMemoryObserver;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;

public class PipelineExperimentSetEvaluator implements IExperimentSetEvaluator, ILoggingCustomizable {

	private Logger logger = LoggerFactory.getLogger("DefaultMetaLearnerExperimentSetEvaluator");

	private PipelineMeasurementConfigContainer container;
	private Timeout timeout;
	private final String executorDetails;

	public PipelineExperimentSetEvaluator(final PipelineMeasurementConfigContainer container, final Timeout timeout, final String executorDetails) {
		this.container = container;
		this.timeout = timeout;
		this.executorDetails = executorDetails;
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, InterruptedException {
		long starttime = System.currentTimeMillis();

		final PeakMemoryObserver mobs = new PeakMemoryObserver();
		mobs.start();

		this.logger.info("Reading in experiment with id {}.", experimentEntry.getId());
		Map<String, String> keys = experimentEntry.getExperiment().getValuesOfKeyFields();
		int seed = Integer.parseInt(keys.get("seed"));
		int openmlid = Integer.parseInt(keys.get("openmlid"));
		int datapoints = Integer.parseInt(keys.get("datapoints"));
		String preprocessor = keys.get("preprocessor");
		String baselearner = keys.get("baselearner");
		String metalearner = keys.get("metalearner");

		this.logger.info("Running experiment {}, which is on dataset {} with seed {} and {} data points. Executor details: {}", experimentEntry.getId(), openmlid, seed, datapoints, this.executorDetails);
		Map<String, Object> map = new HashMap<>();
		map.put("executordetails", this.executorDetails);
		processor.processResults(map);
		map.clear();

		/* load dataset */
		List<ILabeledDataset<?>> splitTmp = null;
		MLPipeline pipeline = null;
		try {
			Classifier baseClassifier = AbstractClassifier.forName(baselearner, null);
			Classifier classifier;
			if (metalearner.trim().isEmpty()) {
				classifier = baseClassifier;
			}
			else {
				classifier = AbstractClassifier.forName(metalearner, null);
				((SingleClassifierEnhancer)classifier).setClassifier(baseClassifier);
			}
			ASSearch searcher = null;
			ASEvaluation evaluator = null;
			if (!preprocessor.trim().isEmpty()) {
				if (preprocessor.contains("/")) {
					String[] parts = preprocessor.split("/");
					evaluator = ASEvaluation.forName(parts[0], null);
					searcher = ASSearch.forName(parts[1].equals("bfs") ? "BestFirst" : "GreedyStepWise", null);
				}
				else {
					evaluator = ASEvaluation.forName(preprocessor, null);
					searcher = new Ranker();
				}
			}
			if (searcher != null) {
				System.out.println(searcher.getClass());
				System.out.println(evaluator.getClass());
			}
			pipeline = new MLPipeline(searcher, evaluator, classifier);

			Dataset ds = (Dataset) OpenMLDatasetReader.deserializeDataset(openmlid);
			if (ds.getLabelAttribute() instanceof INumericAttribute) {
				this.logger.info("Converting numeric dataset to classification dataset!");
				ds = (Dataset) DatasetUtil.convertToClassificationDataset(ds);
			}

			/* check whether the dataset is reproducible */
			if (ds.getConstructionPlan().getInstructions().isEmpty()) {
				mobs.cancel();
				throw new IllegalStateException("Construction plan for dataset is empty!");
			}

			this.logger.info("Label: {} ... {}", ds.getLabelAttribute().getClass().getName(), ds.getLabelAttribute().getStringDescriptionOfDomain());
			if (datapoints >= ds.size()) { // also forbid to use 100% of the data for training (no testing possible)
				throw new IllegalStateException("Dataset has not sufficient datapoints.");
			}
			double portion = datapoints * 1.0 / ds.size();
			splitTmp = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, portion);

		} catch (Throwable e) {
			mobs.cancel();
			throw new ExperimentEvaluationFailedException(e);
		}
		final List<ILabeledDataset<?>> split = splitTmp;
		final IWekaClassifier c = new WekaClassifier(pipeline);

		/* now train classifier */
		long timePriorToTrainCommand = System.currentTimeMillis();
		this.logger.info("Experiment preparaion (including splits) finished after {}s", MathExt.round((timePriorToTrainCommand - starttime) / 1000.0, 2));
		SimpleDateFormat format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
		map.put("train_start", format.format(new Date(System.currentTimeMillis())));
		long deadlinetimestamp = timePriorToTrainCommand + this.timeout.milliseconds();
		try {
			mobs.reset();
			this.logger.info("Starting training for pipeline {}", pipeline);
			TimedComputation.compute(() -> {
				c.fit(split.get(0));
				return null;
			}, this.timeout, "Experiment timeout exceeded.");
			Thread.sleep(1000);
		} catch (Throwable e) {
			map.put("train_end", format.format(new Date(System.currentTimeMillis())));
			processor.processResults(map);
			mobs.cancel();
			throw new ExperimentEvaluationFailedException(e);
		}
		map.put("train_end", format.format(new Date(System.currentTimeMillis())));
		map.put("memory_peak", mobs.getMaxMemoryConsumptionObserved());
		this.logger.info("Finished training, now testing on {} data points. Memory peak was {}", split.get(1).size(), map.get("memory_peak"));
		map.put("test_start", format.format(new Date(System.currentTimeMillis())));
		List<Integer> gt = new ArrayList<>();
		List<Integer> pr = new ArrayList<>();
		DescriptiveStatistics testRuntimeStats = new DescriptiveStatistics();
		try {
			long lastTimeoutCheck = 0;
			int n = split.get(1).size();
			for (ILabeledInstance i : split.get(1)) {
				if (System.currentTimeMillis() - lastTimeoutCheck > 10000) {
					lastTimeoutCheck = System.currentTimeMillis();
					long remainingTime = deadlinetimestamp - lastTimeoutCheck;
					this.logger.debug("Remaining time for this classifier: {}", remainingTime);
					if (remainingTime <= 0) {
						this.logger.info("Triggering timeout ...");
						throw new AlgorithmTimeoutedException(0);
					}
				}
				gt.add((int) i.getLabel());
				long predictionStart = System.currentTimeMillis();
				pr.add((int) c.predict(i).getPrediction());
				testRuntimeStats.addValue(System.currentTimeMillis() - predictionStart);
				if (gt.size() % 10000 == 0) {
					this.logger.info("{}/{} ({}%)", gt.size(), n, gt.size() * 100.0 / n);
				}
				else if (this.logger.isDebugEnabled()) {
					this.logger.debug("{}/{} ({}%)", gt.size(), n, gt.size() * 100.0 / n);
				}
			}
		} catch (Throwable e) {
			map.put("test_end", format.format(new Date(System.currentTimeMillis())));
			processor.processResults(map);
			throw new ExperimentEvaluationFailedException(e);
		}
		map.put("test_end", format.format(new Date(System.currentTimeMillis())));
		map.put("gt", gt);
		map.put("pr", pr);
		this.logger.info("Finished experiment #{}. Updating table. Here is the stats of the test runtimes:\n{}", experimentEntry.getId(), testRuntimeStats);
		processor.processResults(map);
		mobs.cancel();
		this.logger.info("Finished Experiment {}. Results: {}.", experimentEntry.getExperiment().getValuesOfKeyFields(), map);
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}
}
