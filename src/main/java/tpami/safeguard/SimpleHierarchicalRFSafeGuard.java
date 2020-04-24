package tpami.safeguard;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.ICategoricalAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollectionOneLayerPartition;
import ai.libs.mlplan.safeguard.IEvaluationSafeGuard;
import weka.classifiers.trees.J48;

public class SimpleHierarchicalRFSafeGuard implements IEvaluationSafeGuard {

	private static final File PARAMETERIZED_DIRECTORY = new File("python/data/parameterized/");
	private static final String PARAMETERIZED_FILE_NAME_TEMPLATE = "runtimes_%s_parametrized_nooutliers.csv";

	private static final Logger LOGGER = LoggerFactory.getLogger(SimpleHierarchicalRFSafeGuard.class);
	private Lock lock = new ReentrantLock();
	private Map<String, IComponentPredictor> componentRuntimePredictorMap = new HashMap<>();
	private Map<String, IMetaLearnerPredictor> metaLearnerPredictorMap = new HashMap<>();

	public SimpleHierarchicalRFSafeGuard(final File baseDefaultData, final int... excludeOpenMLDatasets) throws Exception {
		/*  Build component predictors */
		// Load data...
		KVStoreCollection baseDefaultCol = DataBasedComponentPredictorUtil.readCSV(baseDefaultData, new HashMap<>());

		// Clean from excluded openml ids
		Map<String, Collection<String>> cleanTrainingData = new HashMap<>();
		cleanTrainingData.put("openmlid", Arrays.stream(excludeOpenMLDatasets).mapToObj(x -> x + "").collect(Collectors.toList()));
		baseDefaultCol.removeAnyContained(cleanTrainingData, true);

		Set<String> ids = baseDefaultCol.stream().map(x -> x.getAsString("openmlid")).collect(Collectors.toSet());
		List<String> idList = new ArrayList<>(ids);
		Collections.sort(idList);
		System.out.println(ids.size() + " " + idList);

		Set<String> components = baseDefaultCol.stream().map(x -> x.getAsString("algorithm")).collect(Collectors.toSet());

		// Partition kvstore collections and fill predictor map
		KVStoreCollectionOneLayerPartition defaultPartition = new KVStoreCollectionOneLayerPartition("algorithm", baseDefaultCol);
		components.parallelStream().forEach(component -> {
			if (!component.equals(J48.class.getName())) {
				return;
			}
			File paramCSV = new File(PARAMETERIZED_DIRECTORY, String.format(PARAMETERIZED_FILE_NAME_TEMPLATE, DataBasedComponentPredictorUtil.mapWeka2ID(component)));
			KVStoreCollection paramCol = null;
			if (paramCSV.exists()) {
				try {
					paramCol = DataBasedComponentPredictorUtil.readCSV(paramCSV, new HashMap<>());
					paramCol.removeAnyContained(cleanTrainingData, true);
				} catch (IOException e) {
					LOGGER.warn("Could not read csv file {}", paramCSV.getAbsolutePath(), e);
				}
			}
			RandomForestComponentPredictor pred = null;
			try {
				pred = new RandomForestComponentPredictor(component, defaultPartition.getData().get(component), paramCol);
			} catch (Exception e) {
				e.printStackTrace();
			}

			this.lock.lock();
			try {
				this.componentRuntimePredictorMap.put(component, pred);
			} finally {
				this.lock.unlock();
			}
		});
		// Build meta learner predictors
	}

	@Override
	public double predictInductionTime(final ComponentInstance ci, final double[] metaFeaturesTrain) throws Exception {
		MLComponentInstanceWrapper ciw;
		if (ci instanceof MLComponentInstanceWrapper) {
			ciw = (MLComponentInstanceWrapper) ci;
		} else {
			ciw = new MLComponentInstanceWrapper(ci);
		}

		if (ciw.isPipeline()) {
			double inductionTime = 0.0;

			// Compute runtime for executing preprocessor
			MLComponentInstanceWrapper preprocessor = ciw.getPreprocessor();
			inductionTime += this.predictInductionTime(preprocessor, metaFeaturesTrain);
			inductionTime += this.predictInferenceTime(preprocessor, metaFeaturesTrain);

			// Compute features for after
			double[] metaFeaturesAfterPP = this.predictMetaFeaturesAfterPreprocessor(preprocessor, metaFeaturesTrain);

			inductionTime += this.predictInductionTime(ciw.getClassifier(), metaFeaturesAfterPP);
			return inductionTime;
		} else if (ciw.isPreprocessor()) {
			String preprocessorIdentifier = DataBasedComponentPredictorUtil.componentInstanceToPreprocessorID(ciw);
			if (preprocessorIdentifier != null && this.componentRuntimePredictorMap.containsKey(preprocessorIdentifier)) {
				return this.componentRuntimePredictorMap.get(preprocessorIdentifier).predictInductionTime(ciw, metaFeaturesTrain);
			} else {
				LOGGER.warn("Unknown component instance for preprocessor: {}", ciw);
				return 0.0;
			}
		} else if (ciw.isMetaLearner()) {
			IMetaLearnerPredictor mlPred = this.metaLearnerPredictorMap.get(ciw.getComponent().getName());
			if (mlPred != null) {
				return mlPred.predictInductionTime(ciw, this.componentRuntimePredictorMap.get(ciw.getBaseLearner().getComponent().getName()), metaFeaturesTrain);
			} else {
				LOGGER.warn("I do not know this meta learner, so I predict a runtime of 0 for {}", ciw);
				return 0.0;
			}
		} else if (ciw.isBaseLearner()) {
			return this.componentRuntimePredictorMap.get(ciw.getComponent().getName()).predictInductionTime(ci, metaFeaturesTrain);
		} else {
			LOGGER.warn("Case not covered. This component instance {} seems to be neither pipeline nor meta learner nor base learner. Therefore we return 0 as time which might be overly enthusiastic.", ci);
			return 0.0;
		}
	}

	@Override
	public double predictInferenceTime(final ComponentInstance ci, final double[] metaFeaturesTest) throws Exception {
		MLComponentInstanceWrapper ciw;
		if (ci instanceof MLComponentInstanceWrapper) {
			ciw = (MLComponentInstanceWrapper) ci;
		} else {
			ciw = new MLComponentInstanceWrapper(ci);
		}

		if (ciw.isPipeline()) {
			double inferenceTime = 0.0;
			MLComponentInstanceWrapper preprocessor = ciw.getPreprocessor();
			inferenceTime += this.predictInferenceTime(preprocessor, metaFeaturesTest);
			double[] metaFeaturesAfterPP = this.predictMetaFeaturesAfterPreprocessor(preprocessor, metaFeaturesTest);
			inferenceTime += this.predictInferenceTime(ciw.getClassifier(), metaFeaturesAfterPP);
			return inferenceTime;
		} else if (ciw.isPreprocessor()) {
			String preprocessorIdentifier = DataBasedComponentPredictorUtil.componentInstanceToPreprocessorID(ciw);
			if (preprocessorIdentifier != null && this.componentRuntimePredictorMap.containsKey(preprocessorIdentifier)) {
				return this.componentRuntimePredictorMap.get(preprocessorIdentifier).predictInferenceTime(ciw, metaFeaturesTest);
			} else {
				LOGGER.warn("Unknown component instance for preprocessor: {}", ciw);
				return 0.0;
			}
		} else if (ciw.isMetaLearner()) {
			IMetaLearnerPredictor mlPred = this.metaLearnerPredictorMap.get(ciw.getComponent().getName());
			if (mlPred != null) {
				return mlPred.predictInferenceTime(ciw, this.componentRuntimePredictorMap.get(ciw.getBaseLearner().getComponent().getName()), metaFeaturesTest);
			} else {
				LOGGER.warn("I do not know this meta learner, so I predict a runtime of 0 for {}", ciw);
				return 0.0;
			}
		} else if (ciw.isBaseLearner()) {
			return this.componentRuntimePredictorMap.get(ciw.getComponent().getName()).predictInferenceTime(ci, metaFeaturesTest);
		} else {
			LOGGER.warn("Case not covered. This component instance {} seems to be neither pipeline nor meta learner nor base learner. Therefore we return 0 as time which might be overly enthusiastic.", ci);
			return 0.0;
		}
	}

	private double[] predictMetaFeaturesAfterPreprocessor(final MLComponentInstanceWrapper preprocessor, final double[] metaFeaturesTrain) {
		// TODO: Implement this method to actually make predictions.
		return metaFeaturesTrain;
	}

	@Override
	public void updateWithActualInformation(final ComponentInstance ci, final double inductionTime, final double inferenceTime) {
		MLComponentInstanceWrapper ciw = new MLComponentInstanceWrapper(ci);
		if (ciw.isBaseLearner()) {
			IComponentPredictor pred = this.componentRuntimePredictorMap.get(ciw.getComponent().getName());
			pred.setActualDefaultConfigurationInductionTime(inductionTime);
			pred.setActualDefaultConfigurationInferenceTime(inferenceTime);
		}
	}

	@Override
	public double[] computeDatasetMetaFeatures(final ILabeledDataset<?> dataset) {
		double numAttributesAfterBinarization = 0.0;
		for (IAttribute attribute : dataset.getListOfAttributes()) {
			if (attribute instanceof INumericAttribute) {
				numAttributesAfterBinarization += 1;
			} else if (attribute instanceof ICategoricalAttribute) {
				numAttributesAfterBinarization += ((ICategoricalAttribute) attribute).getNumberOfCategories();
			}
		}
		return new double[] { dataset.size(), numAttributesAfterBinarization };
	}

}
