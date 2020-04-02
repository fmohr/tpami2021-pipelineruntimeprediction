package tpami.basealgorithmlearning.regression;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.pipeline.featurepreprocess.Standardization;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstancesUtil;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;

public class BaseAlgorithmPerformanceLearner {
	public static void main(final String[] args) throws Exception {

		/* read and split data */
		ILabeledDataset<?> origData = ArffDatasetAdapter.readDataset(new File("bl_logistic.arff"));
		origData.removeIf(i -> (double)i.getLabel() <= 0.5);
		System.out.println("Remaining size: " + origData.size());
		Map<IAttribute, Function<ILabeledInstance, Double>> map = new HashMap<>();
		Pair<List<IAttribute>, Map<IAttribute, Function<ILabeledInstance, Double>>> expansionDescription = DatasetUtil.getPairOfNewAttributesAndExpansionMap(origData, DatasetUtil.EXPANSION_PRODUCTS);
		map.putAll(expansionDescription.getY());
		origData = DatasetUtil.getExpansionOfDataset(origData, expansionDescription);
		Pair<List<IAttribute>, Map<IAttribute, Function<ILabeledInstance, Double>>> squareExpansionDescription = DatasetUtil.getPairOfNewAttributesAndExpansionMap(origData, DatasetUtil.EXPANSION_SQUARES);
		map.putAll(squareExpansionDescription.getY());
		origData = DatasetUtil.getExpansionOfDataset(origData, squareExpansionDescription);
		origData.forEach(i -> i.setLabel((double)i.getLabel() > 0 ? Math.log((double)i.getLabel()) : Math.log(0.05)));
		IWekaInstances data = new WekaInstances(origData);
		if (data.getNumAttributes() != origData.getInstanceSchema().getNumAttributes()) {
			throw new IllegalStateException();
		}
		Instances wekaData = data.getInstances();
		Standardization norm = new Standardization();
		norm.prepare(wekaData);
		wekaData = norm.apply(wekaData);
		wekaData.forEach(i -> System.out.println(Arrays.toString(i.toDoubleArray())));

		/* learn a forest */
		Classifier rf = new MultilayerPerceptron();
		rf.buildClassifier(wekaData);
		System.out.println("Predictor ready.");

		/* test forest on completely different data */
		for (int id : new int[] {
				//				478, 1037,
				//				41946,
				40900,
				41164}) {
			IWekaInstances newDataset = new WekaInstances(OpenMLDatasetReader.deserializeDataset(id));
			Logistic l = new Logistic();
			Map<String, Object> classifierFeatures = BaseAlgorithmDatasetPreparer.getClassifierFeatureRepresentation(l);
			System.out.println("Starting projection");
			for (double i = 0.05; i <= 1; i+= 0.05) {
				Map<String, Object> features = new HashMap<>(classifierFeatures);
				IWekaInstances reducedNewDataset = ((WekaInstances)SplitterUtil.getSimpleTrainTestSplit(newDataset, 0, i).get(0));
				features.putAll(new BasicDatasetFeatureGenerator("d_").getFeatureRepresentation(reducedNewDataset));
				//				features.put("d_instances", Integer.MAX_VALUE);
				ILabeledInstance inst = DatasetUtil.getInstanceFromMap(data.getInstanceSchema(), features, "runtime", map);
				Instance wekaInst = WekaInstancesUtil.transformInstanceToWekaInstance(data.getInstanceSchema(), inst);
				wekaInst = norm.apply(wekaInst);
				wekaInst.setDataset(wekaData);
				System.out.println(Arrays.toString(wekaInst.toDoubleArray()));
				double runtimePrediction = rf.classifyInstance(wekaInst);
				System.out.println("Start learning. Predicted runtime is: " + Math.exp(runtimePrediction) + "s");
				long start = System.currentTimeMillis();
				//				l.buildClassifier(reducedNewDataset.getInstances());
				int trueRuntimeInMS = (int)(System.currentTimeMillis() - start);
				System.out.println(features + ": " + runtimePrediction + ". True time: " + (trueRuntimeInMS / 1000.0));
			}

		}

	}

}
