package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class InfoGainAEOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> OPTIONS_M = Arrays.asList("", "-M");
	private static final List<String> OPTIONS_B = Arrays.asList("", "-B");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(OPTIONS_M, OPTIONS_B)));

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
		options += COMBOS.get(i).get(1);
		return options;
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
