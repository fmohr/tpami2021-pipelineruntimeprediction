package tpami.safeguard;

import java.util.LinkedList;

import org.junit.Test;

import ai.libs.jaicore.basic.sets.Pair;

public class EvaluationTimeCalibrationModuleTest {

	@Test
	public void testCalibrationFactorComputatation() throws Exception {
		EvaluationTimeCalibrationModule calibrator = new EvaluationTimeCalibrationModule(new LinkedList<>());
		Pair<Double, Double> factor = calibrator.getSystemCalibrationFactor();
		System.out.println("Calibration factor: " + factor.getX() + " / " + factor.getY());
	}

}
