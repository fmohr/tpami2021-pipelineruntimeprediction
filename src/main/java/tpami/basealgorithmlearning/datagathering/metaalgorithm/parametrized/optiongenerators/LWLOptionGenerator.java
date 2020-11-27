package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class LWLOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> K = Arrays.asList("0", "1", "10", "100", "1000", "10000", "100000");
	private static final List<String> U = Arrays.asList("0", "1", "2", "3", "4");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(K, U)));

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		StringBuilder sb = new StringBuilder();
		List<String> options = COMBOS.get(i);
		sb.append("-K " + options.get(0));
		sb.append(" -U " + options.get(1));
		return sb.toString();
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
