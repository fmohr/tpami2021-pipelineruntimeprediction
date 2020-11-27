package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized;

import java.util.Collection;
import java.util.List;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.ml.weka.WekaUtil;

public class ParametrizedPreprocessingTableSetup {

	public static void main(final String[] args) throws Exception {

		Collection<List<String>> combos = WekaUtil.getAdmissibleSearcherEvaluatorCombinationsForAttributeSelection();
		System.out.println(combos);
		for (List<String> combo : combos) {


			//			String ss = combo.get(0).toLowerCase();
			//			if (ss.contains("bestfirst") )  {
			//				continue;
			//			}
			//			String es = combo.get(1).toLowerCase();
			//
			//			if (es.contains("infogain") || es.contains("cfssubset") || es.contains("oner") || es.contains("principal") || es.contains("uncert") || es.contains("correlation") || es.contains("gainratio")) {
			//				continue;
			//			}

			/* prepare database for this combination */
			String searcher = Class.forName(combo.get(0)).getSimpleName();
			String evaluator = Class.forName(combo.get(1)).getSimpleName();
			ParametrizedPreprocessorConfigContainer container = new ParametrizedPreprocessorConfigContainer("conf/dbcon-local.conf", searcher, evaluator);
			ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
			//			preparer.setLoggerName("example");
			preparer.synchronizeExperiments();
		}
	}
}
