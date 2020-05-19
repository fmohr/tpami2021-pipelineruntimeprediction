package tpami.safeguard;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.datastructure.kvstore.IKVStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import tpami.safeguard.api.IEvaluationTimeCalibrationModule;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import weka.classifiers.AbstractClassifier;

public class EvaluationTimeCalibrationModule implements IEvaluationTimeCalibrationModule {

	private static final Logger LOGGER = LoggerFactory.getLogger(EvaluationTimeCalibrationModule.class);
	private static final IEvaluationTimeCalibrationModuleConfig CONFIG = ConfigFactory.create(IEvaluationTimeCalibrationModuleConfig.class);

	private final KVStoreCollection baselineData;

	private EventBus eventBus = new EventBus();

	@Override
	public void registerListener(final Object listener) {
		this.eventBus.register(listener);
	}

	public EvaluationTimeCalibrationModule(KVStoreCollection baselineData) throws IOException {
		// prepare baseline data
		List<String> fitSizes = CONFIG.getCalibrationConfigFitSizes();
		List<String> datasetIDs = CONFIG.getCalibrationConfigDatasetIDs();
		switch (CONFIG.getCalibrationConfigMode()) {
		case "crossProduct": {
			Map<String, Collection<String>> containsSelect = new HashMap<>();
			containsSelect.put("fitsize", fitSizes);
			containsSelect.put("openmlid", datasetIDs);
			baselineData = baselineData.selectContained(containsSelect, false);
			break;
		}
		case "pairWise": {
			baselineData.stream().forEach(x -> x.put("pairWiseValue", x.getAsString("fitsize") + "#" + x.getAsString("openmlid")));
			Map<String, Collection<String>> containsSelect = new HashMap<>();
			Set<String> instancesToInclude = new HashSet<>();
			if (fitSizes.size() != datasetIDs.size()) {
				LOGGER.warn("Pair-wise definition of calibration config and the number of fit sizes is not equal to the number of dataste IDs. This leads to ignoring the last entries of the longer list.");
			}
			for (int i = 0; i < Math.min(fitSizes.size(), datasetIDs.size()); i++) {
				instancesToInclude.add(fitSizes.get(i) + "#" + datasetIDs.get(i));
			}
			containsSelect.put("pairWiseValue", instancesToInclude);

			baselineData = baselineData.selectContained(containsSelect, true);
			baselineData.projectRemove("pairWiseValue");
			break;
		}
		}

		if (CONFIG.getEnableBaselineEvaluationTimeFilter()) {
			String collectionID = baselineData.getCollectionID();
			baselineData = new KVStoreCollection(baselineData.stream().filter(x -> {
				if (!x.containsKey("fittime") || !x.containsKey("applicationtime") || x.getAsString("fittime").trim().equals("") || x.getAsString("applicationtime").trim().equals("")) {
					return false;
				}

				int totalEvalTime = x.getAsInt("fittime") + x.getAsInt("applicationtime");

				boolean result = true;
				if (CONFIG.getMinBaselineEvaluationTime() != null) {
					System.out.println("Filter samples by min baseline evalution time " + CONFIG.getMinBaselineEvaluationTime());
					result = result && (totalEvalTime >= CONFIG.getMinBaselineEvaluationTime());
				}
				if (CONFIG.getMaxBaselineEvaluationTime() != null) {
					System.out.println("Filter samples by max baseline evalution time " + CONFIG.getMaxBaselineEvaluationTime());
					result = result && (totalEvalTime <= CONFIG.getMaxBaselineEvaluationTime());
				}
				return result;
			}).collect(Collectors.toList()));
			baselineData.setCollectionID(collectionID);
		}

		this.baselineData = baselineData;
		double min = this.baselineData.stream().mapToDouble(x -> x.getAsDouble("fittime") + x.getAsDouble("applicationtime")).min().getAsDouble();
		double max = this.baselineData.stream().mapToDouble(x -> x.getAsDouble("fittime") + x.getAsDouble("applicationtime")).max().getAsDouble();
		LOGGER.debug("Available samples for calibration: {} with minimum eval time {} and maximum eval time {}", this.baselineData.size(), min, max);
	}

	@Override
	public Pair<Double, Double> getSystemCalibrationFactor() throws Exception {
		int numberOfSamples = Math.min(this.baselineData.size(), CONFIG.getMaxSamples());
		KVStoreCollection samples = new KVStoreCollection();
		if (numberOfSamples < this.baselineData.size()) {
			List<Integer> indices = IntStream.range(0, this.baselineData.size()).mapToObj(x -> x).collect(Collectors.toList());
			Collections.shuffle(indices, new Random(CONFIG.getSeed()));
			IntStream.range(0, CONFIG.getMaxSamples()).mapToObj(x -> this.baselineData.get(indices.get(x))).forEach(samples::add);
		} else {
			samples.addAll(this.baselineData);
		}

		Map<Integer, Dataset> datasetCache = new HashMap<>();
		Lock lock = new ReentrantLock();

		List<IKVStore> evalList = Collections.synchronizedList(new ArrayList<>(samples.size()));

		List<Callable<OptionalDouble>> runnables = new ArrayList<>(numberOfSamples);
		samples.stream().forEach(x -> {
			if (DataBasedComponentPredictorUtil.isPreprocessor(x.getAsString("algorithm"))) {
				return;
			}

			runnables.add(new Callable<OptionalDouble>() {
				@Override
				public OptionalDouble call() throws Exception {
					if (x.getAsInt("fittime") == 0 && x.getAsInt("applicationtime") == 0) {
						return OptionalDouble.empty();
					}
					List<String> skip = Arrays.asList("ReliefFAS", "cfssubseteval_bf");
					if (skip.contains(x.getAsString("algorithm"))) {
						return OptionalDouble.of(1.0);
					}

					IWekaClassifier model = new WekaClassifier(AbstractClassifier.forName(DataBasedComponentPredictorUtil.mapID2Weka(x.getAsString("algorithm")), null));
					int openmlID = x.getAsInt("openmlid");
					int fitSize = x.getAsInt("fitsize");
					int totalSize = x.getAsInt("totalsize");
					int evaluationTime = x.getAsInt("fittime") + x.getAsInt("applicationtime");

					Dataset d = null;
					lock.lock();
					try {
						d = datasetCache.computeIfAbsent(openmlID, t -> {
							try {
								return (Dataset) OpenMLDatasetReader.deserializeDataset(t);
							} catch (DatasetDeserializationFailedException e) {
								e.printStackTrace();
							}
							return null;
						});
					} finally {
						lock.unlock();
					}
					if (d == null) {
						throw new DatasetDeserializationFailedException();
					}

					if (d.getLabelAttribute() instanceof INumericAttribute) {
						d = (Dataset) DatasetUtil.convertToClassificationDataset(d);
					}

					double trainPortion = fitSize * 1.0 / totalSize;
					List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(d, CONFIG.getSeed(), trainPortion);

					long fitStart = System.currentTimeMillis();
					model.fit(split.get(0));
					long fitTime = Math.round((System.currentTimeMillis() - fitStart) * 1.0 / 1000);

					long predictStart = System.currentTimeMillis();
					model.predict(split.get(1));
					long predictTime = Math.round((System.currentTimeMillis() - predictStart) * 1.0);

					IKVStore store = new KVStore();
					store.put("actual-fittime", fitTime);
					store.put("actual-predict", predictTime);
					store.put("actual-eval", fitTime + predictTime);

					store.put("data-fittime", x.getAsInt("fittime"));
					store.put("data-predict", x.getAsInt("applicationtime") * 1000);
					store.put("data-eval", x.getAsInt("fittime") + x.getAsInt("applicationtime"));

					store.put("data-fitsize", fitSize);
					store.put("data-predictsize", totalSize - fitSize);
					store.put("algorithm", x.getAsString("algorithm"));
					evalList.add(store);

					if (evaluationTime > 0) {
						return OptionalDouble.of((double) (fitTime + predictTime) / evaluationTime);
					} else {
						return OptionalDouble.empty();
					}
				}
			});
		});
		LOGGER.debug("Created a list of {} tasks", runnables.size());

		if (CONFIG.getNumCPUs() <= 1) {
			System.out.println("No parallelization start sequentially processing the task queue.");
			runnables.stream().map(x -> {
				try {
					x.call();
				} catch (Exception e) {
					e.printStackTrace();
				}
				return OptionalDouble.empty();
			});
		} else {
			LOGGER.debug("Parallellization is active with {} threads. Create a thread pool and start working", CONFIG.getNumCPUs());
			ForkJoinPool pool = new ForkJoinPool(CONFIG.getNumCPUs());
			List<ForkJoinTask<OptionalDouble>> tasks = runnables.stream().map(x -> pool.submit(x)).collect(Collectors.toList());
			tasks.stream().forEach(x -> {
				try {
					x.get();
				} catch (InterruptedException | ExecutionException e) {
					e.printStackTrace();
				}
			});
			pool.shutdown();
			pool.awaitTermination(1, TimeUnit.HOURS);
		}

		Double fitCalibrationConstant = this.calibrationFactor(evalList.stream().mapToDouble(x -> x.getAsDouble("data-fittime")).toArray(), evalList.stream().mapToDouble(x -> x.getAsDouble("actual-fittime")).toArray());
		Double predictCalibrationConstant = this.calibrationFactor(evalList.stream().mapToDouble(x -> x.getAsDouble("data-predict")).toArray(), evalList.stream().mapToDouble(x -> x.getAsDouble("actual-predict")).toArray());

		evalList.stream().map(x -> {
			double fitError = x.getAsDouble("actual-fittime") - x.getAsDouble("data-fittime") * fitCalibrationConstant;
			double predictError = x.getAsDouble("actual-predict") - x.getAsDouble("data-predict") * predictCalibrationConstant;
			return "Fit Error: " + fitError + " | Predict Error: " + predictError + " | " + x;

		}).forEach(System.out::println);

		this.eventBus.post(new CalibrationConstantsDeterminedEvent(fitCalibrationConstant, predictCalibrationConstant));

		return new Pair<>(fitCalibrationConstant, predictCalibrationConstant);
	}

	private double calibrationFactor(final double[] x, final double[] y) {
		if (x.length != y.length) {
			throw new IllegalArgumentException("x and y must be of the same length.");
		}
		double productSum = IntStream.range(0, x.length).mapToDouble(i -> x[i] * y[i]).sum();
		double squareSum = IntStream.range(0, x.length).mapToDouble(i -> Math.pow(x[i], 2)).sum();
		return productSum / squareSum;
	}

	public void setNumCPUs(final int numCPUs) {
		CONFIG.setProperty(IEvaluationTimeCalibrationModuleConfig.K_CPUS, numCPUs + "");
	}

}
