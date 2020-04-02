package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class JRipOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-E", "-P");
	private static final List<String> O7 = Arrays.asList("1", "2", "3", "4", "5"); // F
	private static final List<String> O8 = Arrays.asList("1", "2", "3", "4", "5"); // N
	private static final List<String> O9 = Arrays.asList("1", "2", "4", "8", "16", "32", "64"); // O

	@Override
	public int getNumberOfValues() {
		return BINARY_OPTIONS.size() + O7.size() + O8.size() + O9.size();
	}

	@Override
	public String getValue(final int i) {
		if (i < BINARY_OPTIONS.size()) {
			return BINARY_OPTIONS.get(i);
		}
		int j = i - BINARY_OPTIONS.size();
		if (j < O7.size()) {
			return "-F " + O7.get(j);
		}
		int k = j - O7.size();
		if (k < O8.size()) {
			return "-N " + O8.get(k);
		}
		return "-O " + O9.get(k - O8.size());
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
