package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class DTOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> O1 = Arrays.asList("", "-I");
	private static final List<String> O2 = Arrays.asList("acc", "rmse", "mae", "auc");
	private static final List<String> O3 = Arrays.asList("weka.attributeSelection.BestFirst", "weka.attributeSelection.GreedyStepwise");
	private static final List<String> O4 = Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8", "9", "10");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(O1, O2, O3, O4)));

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		StringBuilder sb = new StringBuilder();
		List<String> options = COMBOS.get(i);
		sb.append(options.get(0));
		if (sb.length() > 0) {
			sb.append(" ");
		}
		sb.append("-E " + options.get(1));
		sb.append(" -S " + options.get(2));
		sb.append(" -X " + options.get(3));
		return sb.toString();
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
