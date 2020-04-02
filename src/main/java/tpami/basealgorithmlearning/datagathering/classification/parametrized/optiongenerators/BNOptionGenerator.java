package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class BNOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> O1 = Arrays.asList("", "-D");
	private static final List<String> O2 = Arrays.asList("weka.classifiers.bayes.net.search.local.K2", "weka.classifiers.bayes.net.search.local.HillClimber", "weka.classifiers.bayes.net.search.local.LAGDHillClimber", "weka.classifiers.bayes.net.search.local.SimulatedAnnealing", "weka.classifiers.bayes.net.search.local.TabuSearch", "weka.classifiers.bayes.net.search.local.TAN");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(O1, O2)));

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		StringBuilder sb = new StringBuilder();
		List<String> options = COMBOS.get(i);
		sb.append(options.get(0));
		if (sb.length() > 0) {
			sb.append(" ");
		}
		sb.append("-Q " + options.get(1));
		return sb.toString();
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
