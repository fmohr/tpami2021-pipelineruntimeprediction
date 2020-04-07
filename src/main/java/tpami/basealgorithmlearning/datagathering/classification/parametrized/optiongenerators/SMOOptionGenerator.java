package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class SMOOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> C = Arrays.asList("0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100", "1000", "10000");
	private static final List<String> N = Arrays.asList("1", "2");
	private static final List<String> L = Arrays.asList("0.000001", "0.00001", "0.0001", "0.01", "0.1", "1", "10", "100"); // 10^-3 is default and omitted
	private static final List<String> P = Arrays.asList("1.0e-14", "1.0e-13", "1.0e-11", "1.0e-10", "1.0e-9", "1.0e-8", "1.0e-7", "1.0e-6", "1.0e-5", "1.0e-4", "1.0e-3");
	private static final List<String> V = Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8", "9", "10");

	@Override
	public int getNumberOfValues() {
		return C.size() + N.size() + L.size() + P.size() + V.size();
	}

	@Override
	public String getValue(final int i) {
		int j = i;
		if (i < C.size()) {
			return "-C " + C.get(i);
		}
		j -= C.size();
		if (j < N.size()) {
			return "-N " + N.get(j);
		}
		j -= N.size();
		if (j < L.size()) {
			return "-L " + L.get(j);
		}
		j -= L.size();
		if (j < P.size()) {
			return "-P " + P.get(j);
		}
		j -= P.size();
		return "-V " + V.get(j);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
