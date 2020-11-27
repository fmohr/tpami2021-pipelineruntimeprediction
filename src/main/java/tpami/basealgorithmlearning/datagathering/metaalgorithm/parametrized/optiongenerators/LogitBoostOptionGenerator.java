package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class LogitBoostOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> Q = Arrays.asList("", "-Q");
	private static final List<String> L = Arrays.asList("0", "0.01", "0.1");
	private static final List<String> H = Arrays.asList("0.1", "0.5", "0.9");
	private static final List<String> Z = Arrays.asList("1", "2", "3", "5", "10");
	private static final List<String> P = Arrays.asList("50", "60", "70", "80", "90", "95", "100");
	private static final List<String> I = Arrays.asList("5", "10", "20", "50"); // I (default 10 not included)
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(Q, L, H, Z, P, I)));

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
		sb.append("-L " + options.get(1));
		sb.append(" -H " + options.get(2));
		sb.append(" -Z " + options.get(3));
		sb.append(" -P " + options.get(4));
		sb.append(" -I " + options.get(5));
		return sb.toString();
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
