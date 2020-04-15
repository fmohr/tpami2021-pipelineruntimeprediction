package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class ReliefFOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> OPTIONS_K = Arrays.asList("1", "2", "4", "10", "100");
	private static final List<String> OPTIONS_A = Arrays.asList("", "1", "2", "3", "10"); // must be integer
	private static final List<String> OPTIONS_M = Arrays.asList("1", "2", "10", "100", "1000");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(OPTIONS_K, OPTIONS_A, OPTIONS_M))).stream().filter(t -> t.get(1).length() == 0 || (Double.parseDouble(t.get(2)) >= 0.1 * Double.parseDouble(t.get(0)) && Double.parseDouble(t.get(2)) < 0.2 * Double.parseDouble(t.get(0)))).collect(Collectors.toList());

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		String options = "";
		options += "-K " + COMBOS.get(i).get(0);
		if (COMBOS.get(i).get(1).length() > 0) {
			options += " -W -A " + COMBOS.get(i).get(1);
		}
		options += " -M " + COMBOS.get(i).get(2);
		return options;
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
