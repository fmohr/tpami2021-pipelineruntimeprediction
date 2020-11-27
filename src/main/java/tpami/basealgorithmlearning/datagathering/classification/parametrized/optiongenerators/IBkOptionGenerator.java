package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class IBkOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> O1 = Arrays.asList("-K 2", "-K 4", "-K 8", "-K 16", "-K 32", "-K 64"); // -K

	private static final List<String> tuples;

	static {
		try {
			Collection<String> optionCombinations = SetUtil.powerset(Arrays.asList("-X", "-E", "-I")).stream().map(s -> SetUtil.implode(s, " ")).collect(Collectors.toList());
			tuples = new ArrayList<>();
			for (List<String> tuple : SetUtil.cartesianProduct(Arrays.asList(O1, optionCombinations))) {
				tuples.add(tuple.get(0) + " " + tuple.get(1));
			}
		} catch (InterruptedException e) {
			throw new RuntimeException();
		}
	}

	@Override
	public int getNumberOfValues() {
		return tuples.size();
	}

	@Override
	public String getValue(final int i) {
		return tuples.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
