package tpami.safeguard.api;

import ai.libs.jaicore.basic.sets.Pair;

public interface IEvaluationTimeCalibrationModule {

	public Pair<Double, Double> getSystemCalibrationFactor() throws Exception;

}
