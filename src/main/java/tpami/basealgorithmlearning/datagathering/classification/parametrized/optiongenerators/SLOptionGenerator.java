package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class SLOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-S", "-A", "-P");
	private static final List<String> W = Arrays.asList("0", "0.5", "1.0", "1.5", "2.0");
	private static final List<String> H = Arrays.asList("1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024");
	private static final List<String> I = Arrays.asList("1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024");
	private static final List<String> M = Arrays.asList("1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024");

	@Override
	public int getNumberOfValues() {
		return BINARY_OPTIONS.size() + W.size() + H.size() + I.size() + M.size();
	}

	@Override
	public String getValue(final int i) {
		if (i < BINARY_OPTIONS.size()) {
			return BINARY_OPTIONS.get(i);
		}
		int j = i - BINARY_OPTIONS.size();
		if (j < W.size()) {
			return "-W " + W.get(j);
		}
		j -= W.size();
		if (j < H.size()) {
			return "-H " + H.get(j);
		}
		j -= H.size();
		if (j < I.size()) {
			return "-I " + I.get(j);
		}
		j -= I.size();
		return "-M " + M.get(j);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
