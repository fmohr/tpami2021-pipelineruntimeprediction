package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class NBOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-K", "-D");

	@Override
	public int getNumberOfValues() {
		return BINARY_OPTIONS.size();
	}

	@Override
	public String getValue(final int i) {
		return BINARY_OPTIONS.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
