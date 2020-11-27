package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class NBOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-K", "-D");
	private static final List<String> COMBOS = new ArrayList<>();

	static {
		try {
			COMBOS.addAll(SetUtil.powerset(BINARY_OPTIONS).stream().map(l -> SetUtil.implode(l, " ")).collect(Collectors.toList()));
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

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
