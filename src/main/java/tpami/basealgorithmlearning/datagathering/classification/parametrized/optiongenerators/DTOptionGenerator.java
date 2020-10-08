package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class DTOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> O1 = Arrays.asList("", "-I");
	private static final List<String> O2 = Arrays.asList("-E acc", "-E rmse", "-E mae", "-E auc");
	private static final List<String> O3 = Arrays.asList("-S weka.attributeSelection.BestFirst", "-S weka.attributeSelection.GreedyStepwise");
	private static final List<String> O4 = Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8", "9", "10").stream().map(s -> "-X " + s).collect(Collectors.toList());
	private static final List<List<String>> COMBOS;

	static {
		COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(O1, O2, O3, O4)));
	}

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		return SetUtil.implode(COMBOS.get(i), " ");
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
