package test;

import java.util.List;

import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class PipelineConsistencyTest {

	public static void main(final String[] args) throws Exception {
		Instances data = new WekaInstances(OpenMLDatasetReader.deserializeDataset(4541)).getInstances();
		for (int seed = 0; seed < 10; seed ++) {
			List<Instances> split = WekaUtil.getStratifiedSplit(data, seed, 10000.0 / data.size());
			AdaBoostM1 adaBoost = new AdaBoostM1();
			adaBoost.setClassifier(new IBk());
			//			MLPipeline pl = new MLPipeline(new ArrayList<>(), adaBoost);
			long start = System.currentTimeMillis();
			adaBoost.buildClassifier(split.get(0));
			System.out.println("PL: " + (System.currentTimeMillis() - start));
		}
	}
}
