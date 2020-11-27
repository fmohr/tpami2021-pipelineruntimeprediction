package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class BaggingOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> O = Arrays.asList("", "-O");
	private static final List<String> P = Arrays.asList("50", "60", "70", "80", "90", "95");
	private static final List<String> I = Arrays.asList("5", "20", "50"); // I (default 10 not included)
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(O, P, I)));

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
		sb.append("-P " + options.get(1));
		sb.append(" -I " + options.get(2));
		return sb.toString();
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
