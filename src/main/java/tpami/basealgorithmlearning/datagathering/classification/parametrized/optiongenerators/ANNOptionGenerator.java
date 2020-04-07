package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class ANNOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-B", "-R", "-C", "-D");
	private static final List<String> L = Arrays.asList("0.1", "0.2", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"); // L (default 0.3 not included)
	private static final List<String> M = Arrays.asList("0.1", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"); // M (default 0.2 not included)
	private static final List<String> H = Arrays.asList("i", "o", "t"); // W (a not here, because it is already included in the default)

	@Override
	public int getNumberOfValues() {
		return BINARY_OPTIONS.size() + L.size() + M.size() + H.size();
	}

	@Override
	public String getValue(final int i) {
		if (i < BINARY_OPTIONS.size()) {
			return BINARY_OPTIONS.get(i);
		}
		int j = i - BINARY_OPTIONS.size();
		if (j < L.size()) {
			return "-L " + L.get(j);
		}
		j -= L.size();
		if (j < M.size()) {
			return "-M " + M.get(j);
		}
		j -= M.size();
		return "-H " + H.get(j);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
