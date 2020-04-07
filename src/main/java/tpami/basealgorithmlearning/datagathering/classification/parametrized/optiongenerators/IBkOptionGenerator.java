package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class IBkOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> O1 = Arrays.asList("2", "4", "8", "16", "32", "64"); // -K
	private static final List<String> BINARY_OPTIONS = Arrays.asList("-X", "-E", "-I", "-F");

	@Override
	public int getNumberOfValues() {
		return O1.size() + BINARY_OPTIONS.size();
	}

	@Override
	public String getValue(final int i) {
		if (i < BINARY_OPTIONS.size()) {
			return BINARY_OPTIONS.get(i);
		}
		else {
			return "-K " + O1.get(i - BINARY_OPTIONS.size());
		}
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
