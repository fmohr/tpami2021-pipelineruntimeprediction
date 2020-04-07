package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class REPOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-P");
	private static final List<String> M = Arrays.asList("2", "4", "8", "16", "32", "64", "128"); // M: minimum instances per leaf (1 is default)
	private static final List<String> V = Arrays.asList("0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100"); // V: min variance per split (10^-3 is default)
	private static final List<String> L = Arrays.asList("1", "2", "4", "8", "16", "32", "64"); // L: max depth of trees (-1 (no limit) is default)
	private static final List<String> N = Arrays.asList("1", "2", "4", "8", "16", "32", "64"); // N: Number of folds for pruning (3 is default)

	@Override
	public int getNumberOfValues() {
		return BINARY_OPTIONS.size() + M.size() + V.size() + L.size() + N.size();
	}

	@Override
	public String getValue(final int i) {
		if (i < BINARY_OPTIONS.size()) {
			return BINARY_OPTIONS.get(i);
		}
		int j = i - BINARY_OPTIONS.size();
		if (j < M.size()) {
			return "-M " + M.get(j);
		}
		j -= M.size();
		if (j < V.size()) {
			return "-V " + V.get(j);
		}
		j -= V.size();
		if (j < L.size()) {
			return "-L " + L.get(j);
		}
		j -= L.size();
		return "-N " + N.get(j);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
