package tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators;

import java.util.Arrays;
import java.util.List;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class OneROptionGenerator implements IExperimentKeyGenerator<String> {

	private static final List<String> MIN_BUCKET_SIZES = Arrays.asList("1", "2", "4", "6", "8", "16", "32", "64"); // default is 6 (not included here)

	@Override
	public int getNumberOfValues() {
		return MIN_BUCKET_SIZES.size();
	}

	@Override
	public String getValue(final int i) {
		return "-B " + MIN_BUCKET_SIZES.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
