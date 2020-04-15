package tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class CFSSubsetOptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> OPTIONS_M = Arrays.asList("", "-M");
	private static final List<String> OPTIONS_L = Arrays.asList("", "-L");
	private static final List<String> OPTIONS_Z = Arrays.asList("", "-Z");
	private static final List<List<String>> COMBOS = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(OPTIONS_M, OPTIONS_L, OPTIONS_Z)));

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		String options = "";
		for (int j = 0; j < 3; j++) {
			String opt = COMBOS.get(i).get(j);
			if (opt.length() > 0) {
				if (options.length() > 0) {
					options += " ";
				}
				options += opt;
			}
		}
		return options;
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
