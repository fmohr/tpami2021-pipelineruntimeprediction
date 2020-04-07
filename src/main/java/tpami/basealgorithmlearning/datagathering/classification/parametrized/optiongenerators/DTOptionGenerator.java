package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class DTOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> O1 = Arrays.asList("-I");
	private static final List<String> O2 = Arrays.asList("acc", "rmse", "mae", "auc");
	private static final List<String> O3 = Arrays.asList("weka.attributeSelection.BestFirst", "weka.attributeSelection.GreedyStepwise");
	private static final List<String> O4 = Arrays.asList("1", "2", "3", "4", "5", "6", "7", "8", "9", "10");

	@Override
	public int getNumberOfValues() {
		return 1 + 4 + 1 + 10;
	}

	@Override
	public String getValue(final int i) {
		if (i < 1) {
			return O1.get(0);
		}
		if (i < 5) {
			return "-E " + O2.get(i - 1);
		}
		if (i < 6) {
			return "-S " + O3.get(1);
		}
		return "-X " + O4.get(i - 6);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
