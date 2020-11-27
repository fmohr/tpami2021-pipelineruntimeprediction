package tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class RandomCommitteeOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> I = Arrays.asList("5", "10", "20", "50"); // I (default 10 not included)

	@Override
	public int getNumberOfValues() {
		return I.size();
	}

	@Override
	public String getValue(final int i) {
		StringBuilder sb = new StringBuilder();
		sb.append("-I " + I.get(i));
		return sb.toString();
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
