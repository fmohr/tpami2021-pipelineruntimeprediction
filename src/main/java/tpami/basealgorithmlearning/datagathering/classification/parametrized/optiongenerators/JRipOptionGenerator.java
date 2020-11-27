package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class JRipOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-E", "-P");
	private static final List<String> O7 = Arrays.asList("1", "2", "3", "4", "5").stream().map(o -> "-F " + o).collect(Collectors.toList()); // F
	private static final List<String> O8 = Arrays.asList("1", "2", "3", "4", "5").stream().map(o -> "-N " + o).collect(Collectors.toList()); // N
	private static final List<String> O9 = Arrays.asList("1", "2", "4", "8", "16", "32", "64").stream().map(o -> "-O " + o).collect(Collectors.toList()); // O
	private static final List<String> COMBOS = new ArrayList<>();

	@Override
	public int getNumberOfValues() {
		Collection<String> BINARY_COMBOS;
		try {
			BINARY_COMBOS = SetUtil.powerset(BINARY_OPTIONS).stream().map(l -> SetUtil.implode(l, " ")).collect(Collectors.toList());
			COMBOS.addAll(SetUtil.getSubGridRelationFromRelation(SetUtil.cartesianProduct(Arrays.asList(BINARY_COMBOS, O7, O8, O9)), 100).stream().map(c -> SetUtil.implode(c, " ").trim()).collect(Collectors.toList()));
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		return COMBOS.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
