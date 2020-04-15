package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class SymmetricalAttributeOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> OPTIONS_M = Arrays.asList("", "-M");

	@Override
	public int getNumberOfValues() {
		return OPTIONS_M.size();
	}

	@Override
	public String getValue(final int i) {
		String options = "";
		options += OPTIONS_M.get(i);
		return options;
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
