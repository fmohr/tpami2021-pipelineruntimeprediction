package tpami.basealgorithmlearning.datagathering.preprocessing;

import java.util.Collection;
import java.util.List;

import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.ml.weka.WekaUtil;

public class PreprocessingTableSetup {

	public static void main(final String[] args) throws Exception {

		Collection<List<String>> combos = WekaUtil.getAdmissibleSearcherEvaluatorCombinationsForAttributeSelection();
		System.out.println(combos);
		for (List<String> combo : combos) {

			if (!(
					//					combo.get(1).toLowerCase().contains("correlation") || combo.get(1).contains("GainRatioAttributeEval") ||
					combo.get(1).contains("PrincipalComponent")) ) {
				continue;
			}

			/* prepare database for this combination */
			String searcher = Class.forName(combo.get(0)).getSimpleName();
			String evaluator = Class.forName(combo.get(1)).getSimpleName();
			PreprocessorConfigContainer container = new PreprocessorConfigContainer("conf/dbcon-local.conf", searcher, evaluator);
			ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(container.getConfig(), container.getDatabaseHandle());
			preparer.setLoggerName("example");
			preparer.synchronizeExperiments();
		}
	}
}
