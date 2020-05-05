package tpami.safeguard.impl;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import tpami.safeguard.api.IEvaluationTimeCalibrationModule;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import weka.classifiers.AbstractClassifier;

public class EvaluationTimeCalibrationModule implements IEvaluationTimeCalibrationModule {

	private static final Logger LOGGER = LoggerFactory.getLogger(EvaluationTimeCalibrationModule.class);
	private static final List<String> PREPROCESSORS = Arrays.asList("bf/cfssubseteval", "gsw/cfssubseteval", "correlationAS", "PCAAS", "ReliefAS", "GainRatioAS", "InfoGainAS", "SymmetricalUncertAS");

	private final int numCPUs;
	private final int maxSamples;
	private final long seed;
	private final KVStoreCollection baselineData;

	public EvaluationTimeCalibrationModule(final int numCPUs, final int maxSamples, final long seed, final File baseline, final Collection<String> fitSizes, final Collection<String> openMLIDsToConsider) throws IOException {
		// store configurations
		this.numCPUs = numCPUs;
		this.maxSamples = maxSamples;
		this.seed = seed;

		// prepare baseline data
		KVStoreCollection baselineData = DataBasedComponentPredictorUtil.readCSV(baseline, new HashMap<>());
		Map<String, Collection<String>> containsSelect = new HashMap<>();
		containsSelect.put("fitsize", fitSizes);
		containsSelect.put("openmlid", openMLIDsToConsider);
		baselineData = baselineData.selectContained(containsSelect, false);

		// remove preprocessors form samples
		Map<String, Collection<String>> removeCondition = new HashMap<>();
		removeCondition.put("algorithm", PREPROCESSORS);
		baselineData.removeAnyContained(removeCondition, true);

		this.baselineData = baselineData;
		LOGGER.debug("Available samples for calibration: {}", this.baselineData.size());
	}

	@Override
	public double getSystemCalibrationFactor() throws Exception {
		int numberOfSamples = Math.min(this.baselineData.size(), this.maxSamples);
		KVStoreCollection samples = new KVStoreCollection();
		if (numberOfSamples < this.baselineData.size()) {
			List<Integer> indices = IntStream.range(0, this.baselineData.size()).mapToObj(x -> x).collect(Collectors.toList());
			Collections.shuffle(indices, new Random(this.seed));
			IntStream.range(0, this.maxSamples).mapToObj(x -> this.baselineData.get(indices.get(x))).forEach(samples::add);
		} else {
			samples.addAll(this.baselineData);
		}

		List<Callable<OptionalDouble>> runnables = new ArrayList<>(numberOfSamples);
		samples.stream().map(x -> new Callable<OptionalDouble>() {
			@Override
			public OptionalDouble call() throws Exception {
				if (x.getAsInt("fittime") == 0 && x.getAsInt("applicationtime") == 0) {
					return OptionalDouble.empty();
				}
				IWekaClassifier model = new WekaClassifier(AbstractClassifier.forName(x.getAsString("algorithm"), null));
				int openmlID = x.getAsInt("openmlid");
				int fitSize = x.getAsInt("fitsize");
				int totalSize = x.getAsInt("totalsize");
				int evaluationTime = x.getAsInt("fittime") + x.getAsInt("applicationtime");

				Dataset d = (Dataset) OpenMLDatasetReader.deserializeDataset(openmlID);
				double trainPortion = fitSize * 1.0 / totalSize;
				List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(d, EvaluationTimeCalibrationModule.this.seed, trainPortion);
				System.out.println(split.get(0).size() + " " + split.get(1).size());

				long fitStart = System.currentTimeMillis();
				model.fit(split.get(0));
				long fitTime = Math.round((System.currentTimeMillis() - fitStart) * 1.0 / 1000);

				long predictStart = System.currentTimeMillis();
				model.predict(split.get(1));
				long predictTime = Math.round((System.currentTimeMillis() - predictStart) * 1.0 / 1000);

				if (evaluationTime > 0) {
					return OptionalDouble.of((double) (fitTime + predictTime) / evaluationTime);
				} else {
					return OptionalDouble.empty();
				}
			}
		}).forEach(runnables::add);
		LOGGER.debug("Created a list of {} tasks", runnables.size());

		final List<OptionalDouble> results;
		if (this.numCPUs <= 1) {
			System.out.println("No parallelization start sequentially processing the task queue.");
			results = runnables.stream().map(x -> {
				try {
					return x.call();
				} catch (Exception e) {
					e.printStackTrace();
				}
				return OptionalDouble.empty();
			}).collect(Collectors.toList());
		} else {
			LOGGER.debug("Parallellization is active with {} threads. Create a thread pool and start working", this.numCPUs);
			ForkJoinPool pool = new ForkJoinPool(this.numCPUs);
			List<ForkJoinTask<OptionalDouble>> tasks = runnables.stream().map(x -> pool.submit(x)).collect(Collectors.toList());
			results = tasks.stream().map(x -> {
				try {
					return x.get();
				} catch (InterruptedException | ExecutionException e) {
					e.printStackTrace();
					return OptionalDouble.empty();
				}
			}).collect(Collectors.toList());
			pool.shutdown();
		}

		OptionalDouble result = results.stream().filter(x -> x.isPresent()).mapToDouble(x -> x.getAsDouble()).average();
		if (!result.isPresent()) {
			throw new IllegalStateException("Could not determine a calibration factor. List of estimated calibration factors is empty.");
		} else {
			return result.getAsDouble();
		}
	}

}
