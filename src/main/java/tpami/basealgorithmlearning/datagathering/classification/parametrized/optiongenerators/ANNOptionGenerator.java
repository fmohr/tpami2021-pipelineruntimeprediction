package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class ANNOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> BINARY_OPTIONS = Arrays.asList("-B", "-R", "-C", "-D");
	private static final List<String> L = Arrays.asList("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0").stream().map(o -> "-L " +o).collect(Collectors.toList()); // L (default 0.3 not included)
	private static final List<String> M = Arrays.asList("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0").stream().map(o -> "-M " +o).collect(Collectors.toList()); // M (default 0.2 not included)
	private static final List<String> H = Arrays.asList("a", "i", "o", "t").stream().map(o -> "-H " +o).collect(Collectors.toList()); // H (a not here, because it is already included in the default)

	private static final List<String> COMBOS = new ArrayList<>();

	static {
		try {
			Collection<String> BINARY_COMBOS = SetUtil.powerset(BINARY_OPTIONS).stream().map(l -> SetUtil.implode(l, " ")).collect(Collectors.toList());
			COMBOS.addAll(SetUtil.getSubGridRelationFromRelation(SetUtil.cartesianProduct(Arrays.asList(BINARY_COMBOS, L, M, H)), 100).stream().map(c -> SetUtil.implode(c, " ").trim()).collect(Collectors.toList()));
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
