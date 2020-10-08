package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class RankerOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> COMBOS = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).stream().map(n -> "-N " + n).collect(Collectors.toList());

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		return COMBOS.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		return COMBOS.contains(value);
	}

}
