package tpami.basealgorithmlearning;

import ai.libs.jaicore.experiments.IExperimentKeyGenerator;

public class EmptyOptionGenerator implements IExperimentKeyGenerator<String>  {

	@Override
	public int getNumberOfValues() {
		return 0;
	}

	@Override
	public String getValue(final int i) {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isValueValid(final String value) {
		return false;
	}

}
