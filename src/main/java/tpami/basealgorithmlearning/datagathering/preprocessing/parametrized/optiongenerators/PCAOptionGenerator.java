package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class PCAOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> OPTIONS_A = Arrays.asList("-1", "1", "2", "4", "8", "10", "100");
	private static final List<String> OPTIONS_C = Arrays.asList("", "-C");
	private static final List<String> OPTIONS_R = Arrays.asList(".5", ".7", ".9", ".95", ".99");
	private static final List<String> OPTIONS_O = Arrays.asList("", "-O");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(OPTIONS_A, OPTIONS_C, OPTIONS_R, OPTIONS_O)));

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		String options = "";
		options += "-A " + COMBOS.get(i).get(0);
		if (COMBOS.get(i).get(1).length() > 0) {
			options += " " + COMBOS.get(i).get(1);
		}
		options += " -R " + COMBOS.get(i).get(2);
		if (COMBOS.get(i).get(3).length() > 0) {
			options += " " + COMBOS.get(i).get(3);
		}
		return options;
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
