package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class LMTOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-B", "-R", "-C", "-P", "-A");
	private static final List<String> O1 = Arrays.asList("1", "2", "4", "8", "16", "32", "64"); // M
	private static final List<String> O2 = Arrays.asList("0", "0.5", "1", "1.5", "2", "4"); // W

	@Override
	public int getNumberOfValues() {
		return BINARY_OPTIONS.size() + O1.size() + O2.size();
	}

	@Override
	public String getValue(final int i) {
		if (i < BINARY_OPTIONS.size()) {
			return BINARY_OPTIONS.get(i);
		}
		int j = i - BINARY_OPTIONS.size();
		if (j < O1.size()) {
			return "-M " + O1.get(j);
		}
		int k = j - O1.size();
		return "-W " + O2.get(k);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
