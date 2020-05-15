package test;

import java.io.File;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.junit.Test;

import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import weka.classifiers.lazy.KStar;
import weka.core.Instances;

public class MyTest {

	@Test
	public void test() throws Exception {
		ILabeledDataset<?> datasets = ArffDatasetAdapter.readDataset(new File("../datasets/classification/mlj/yeast.arff"));

		Instances dataset = new WekaInstances(datasets).getInstances();
		System.out.println(new Instances(dataset, 0));

		KStar kstar = new KStar();
		kstar.buildClassifier(dataset);

	}
}
