package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class PartOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-R", "-B", "-U", "-J");
	private static final List<String> O1 = Arrays.asList("1", "4", "8", "16", "32", "64"); // M: minimum number of objects per leaf (2 is default)
	private static final List<String> O2 = Arrays.asList("1", "2", "4", "5", "6", "7", "8", "9", "10"); // N: number of folds (3 is default)

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
		j -= O1.size();
		return "-N " + O2.get(j);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
