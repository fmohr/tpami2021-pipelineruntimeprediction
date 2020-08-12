package tpami;

import java.util.Collection;

import ai.libs.jaicore.experiments.ExperimentUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;

public class SQLQueryGenerator {

	public static void main(final String[] args)  {

		System.out.println(ExperimentUtil.getOccurredExceptions("evaluations_classifiers_multilayerperceptron", "AlgorithmTime", "too large"));

		Collection<String> classifiers = WekaUtil.getBasicLearners();
		for (String c : classifiers) {
			String cName = c.substring(c.lastIndexOf(".") + 1).toLowerCase();
			String tablename = "evaluations_classifiers_" + cName;
			System.out.println(cName + ": " + ExperimentUtil.getProgressQuery(tablename));
		}
	}
}
