package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class VPOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> I = Arrays.asList("2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"); // default is 1 (and ommited)
	private static final List<String> E = Arrays.asList("2", "3", "4", "5"); // default is 1 (and ommited)
	private static final List<String> M = Arrays.asList("1", "10", "100", "1000", "100000", "1000000"); // default is 10k (and ommited)

	@Override
	public int getNumberOfValues() {
		return I.size() + E.size() + M.size();
	}

	@Override
	public String getValue(final int i) {
		int j = i;
		if (i < I.size()) {
			return "-I " + I.get(i);
		}
		j -= I.size();
		if (j < E.size()) {
			return "-E " + E.get(j);
		}
		j -= E.size();
		return "-M " + M.get(j);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
