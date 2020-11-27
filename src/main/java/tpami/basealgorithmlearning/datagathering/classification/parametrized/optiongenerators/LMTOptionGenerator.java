package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class LMTOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-B", "-R", "-C", "-P", "-A");
	private static final List<String> O1 = Arrays.asList("1", "2", "4", "8", "16", "32", "64").stream().map(o -> "-M " + o).collect(Collectors.toList()); // M
	private static final List<String> O2 = Arrays.asList("0", "0.5", "1", "1.5", "2", "4").stream().map(o -> "-W " + o).collect(Collectors.toList()); // W

	private static final List<String> COMBOS = new ArrayList<>();

	@Override
	public int getNumberOfValues() {
		Collection<String> BINARY_COMBOS;
		try {
			BINARY_COMBOS = SetUtil.powerset(BINARY_OPTIONS).stream().map(l -> SetUtil.implode(l, " ")).collect(Collectors.toList());
			COMBOS.addAll(SetUtil.getSubGridRelationFromRelation(SetUtil.cartesianProduct(Arrays.asList(BINARY_COMBOS, O1, O2)), 100).stream().map(c -> SetUtil.implode(c, " ").trim()).collect(Collectors.toList()));
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
