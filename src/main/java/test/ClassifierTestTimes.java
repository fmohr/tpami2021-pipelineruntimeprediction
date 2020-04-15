package test;

import java.util.List;

import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.DecisionStump;
import weka.core.Instance;
import weka.core.Instances;

public class ClassifierTestTimes {
	public static void main(final String[] args) throws Exception {
		Instances data = new WekaInstances(OpenMLDatasetReader.deserializeDataset(23512)).getInstances();
		for (int seed = 0; seed < 10; seed ++) {
			List<Instances> split = WekaUtil.getStratifiedSplit(data, seed, 1000.0 / data.size());
			Classifier c = new DecisionStump();
			//			MLPipeline pl = new MLPipeline(new ArrayList<>(), adaBoost);
			long start = System.currentTimeMillis();
			c.buildClassifier(split.get(0));
			long medium = System.currentTimeMillis();
			System.out.println("Training took " + (medium - start) + ". Now classifying " + split.get(1).size() + " instances.");
			for (Instance i : split.get(1)) {
				c.distributionForInstance(i);
			}
			long end = System.currentTimeMillis();
			System.out.println("PT: " + (end - medium));
		}
	}
}
