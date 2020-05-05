package tpami.safeguard;

import java.io.File;
import java.util.Arrays;

import org.junit.Test;

import tpami.safeguard.impl.EvaluationTimeCalibrationModule;

public class EvaluationTimeCalibrationModuleTest {

	@Test
	public void testCalibrationFactorComputatationSequential() throws Exception {
		EvaluationTimeCalibrationModule calibrator = new EvaluationTimeCalibrationModule(1, 10, 42, new File("python/data/runtimes_all_default_nooutliers.csv"), Arrays.asList("16"), Arrays.asList("41066"));
		double factor = calibrator.getSystemCalibrationFactor();
		System.out.println("Calibration factor: " + factor);
	}

	@Test
	public void testCalibrationfactorComputationParallel() throws Exception {
		EvaluationTimeCalibrationModule calibrator = new EvaluationTimeCalibrationModule(4, 10, 42, new File("python/data/runtimes_all_default_nooutliers.csv"), Arrays.asList("16"), Arrays.asList("41066"));
		double factor = calibrator.getSystemCalibrationFactor();
		System.out.println("Calibration factor: " + factor);

	}

}
