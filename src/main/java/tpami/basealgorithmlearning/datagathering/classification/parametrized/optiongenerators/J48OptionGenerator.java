package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class J48OptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-O", "-U", "-B", "J", "-S", "-A");
	private static final List<String> O8 = Arrays.asList("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"); // C
	private static final List<String> O9 = Arrays.asList("1", "4", "8", "16", "32", "64"); // M

	@Override
	public int getNumberOfValues() {
		return BINARY_OPTIONS.size() + O8.size() + O9.size();
	}

	@Override
	public String getValue(final int i) {
		if (i < BINARY_OPTIONS.size()) {
			return BINARY_OPTIONS.get(i);
		}
		int j = i - BINARY_OPTIONS.size();
		if (j < O8.size()) {
			return "-C " + O8.get(j);
		}
		int k = j - O8.size();
		return "-M " + O9.get(k);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
