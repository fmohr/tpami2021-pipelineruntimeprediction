package tpami.safeguard;

import org.junit.Test;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.sets.Pair;

public class EvaluationTimeCalibrationModuleTest {

	@Test
	public void testCalibrationFactorComputatation() throws Exception {
		EvaluationTimeCalibrationModule calibrator = new EvaluationTimeCalibrationModule(new KVStoreCollection());
		Pair<Double, Double> factor = calibrator.getSystemCalibrationFactor();
		System.out.println("Calibration factor: " + factor.getX() + " / " + factor.getY());
	}

}
