package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class OneRAEOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> OPTIONS_D = Arrays.asList("", "-D");
	private static final List<String> OPTIONS_F = Arrays.asList("2", "4", "8", "10");
	private static final List<String> OPTIONS_B = Arrays.asList("1", "2", "4", "6", "8", "16");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(OPTIONS_D, OPTIONS_F, OPTIONS_B)));

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		String options = "";
		options += COMBOS.get(i).get(0);
		if (options.length() > 0) {
			options += " ";
		}
		options += "-F " + COMBOS.get(i).get(1);
		options += " -B " + COMBOS.get(i).get(2);
		return options;
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
