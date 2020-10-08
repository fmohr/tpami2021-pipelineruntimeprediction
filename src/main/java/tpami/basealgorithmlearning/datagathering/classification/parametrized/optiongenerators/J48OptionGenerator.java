package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class J48OptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-O", "-U", "-B", "-J", "-S", "-A");
	private static final List<String> O8 = Arrays.asList("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0").stream().map(c -> "-C " + c).collect(Collectors.toList()); // C
	private static final List<String> O9 = Arrays.asList("1", "4", "8", "16", "32", "64").stream().map(c -> "-M " + c).collect(Collectors.toList()); // M
	private static final List<String> COMBOS = new ArrayList<>();

	static {
		try {
			Collection<String> BINARY_COMBOS = SetUtil.powerset(BINARY_OPTIONS).stream().map(l -> SetUtil.implode(l, " ")).collect(Collectors.toList());
			COMBOS.addAll(SetUtil.getSubGridRelationFromRelation(SetUtil.cartesianProduct(Arrays.asList(BINARY_COMBOS, O8, O9)), 100).stream().map(c -> SetUtil.implode(c, " ").trim()).collect(Collectors.toList()));
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
		return true;
	}
}
