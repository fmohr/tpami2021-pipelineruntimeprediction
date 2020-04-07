package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class LogisticOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> RIDGE = Arrays.asList("0.000000001", "0.00000001", "0.0000001", "0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "0", "1.0", "10", "100");

	@Override
	public int getNumberOfValues() {
		return RIDGE.size();
	}

	@Override
	public String getValue(final int i) {
		return "-R " + RIDGE.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
