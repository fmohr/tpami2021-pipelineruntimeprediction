package tpami;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

import ai.libs.jaicore.logging.LoggerUtil;
import tpami.basealgorithmlearning.datagathering.ExperimentUtil;
import tpami.basealgorithmlearning.datagathering.ILearnerExperimentConfig;

public class DatasetProvider {

	public static void main(final String[] args) {
		File confFile = new File(args[0]);
		if (!confFile.exists()) {
			throw new IllegalArgumentException("First argument should be the name of the config file, but the file " + confFile + " could not be found!");
		}
		List<Integer> ids = ((ILearnerExperimentConfig)ConfigFactory.create(ILearnerExperimentConfig.class).loadPropertiesFromFile(confFile)).openMLIDs();
		Objects.requireNonNull(ids, "Could not load ids. Check the config file " + confFile + " and that the field \"openmlids\" is defined.");
		ExperimentUtil util = new ExperimentUtil();
		util.setLoggerName(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);
		if  (args.length > 0) {
			final int index = Integer.parseInt(args[1]);
			final int id = ids.get(index);
			ids.removeIf(i -> i != id);
		}

		System.out.println("Iterating over ids: " + ids);
		ids.forEach(id -> {

			try {
				util.createSMOTEExtensionOfOpenMLDatasets(id);
			} catch (Exception e) {
				e.printStackTrace();
			}
			System.out.println("Finished SMOTE process for all limits on dataset " + id);
		});
		System.out.println("Finished SMOTE process. Now checking that we can produce datasets of the desired forms for different sizes.");

		ids.forEach(id -> {
			for (int expectedTrainSize : Arrays.asList(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 15000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 250000, 500000, 750000, 1000000)) {
				int expectedTestSize = 1500;
				int numAttributes = Math.min(40000, (int)Math.ceil(3.0 * Math.pow(10, 8) / expectedTrainSize));
				System.out.println("Checking for train size " + expectedTrainSize + ". Max relevant attribute size is " + numAttributes);
				try {
					List<ILabeledDataset<?>> split = util.createSizeAdjustedTrainTestSplit(id, expectedTrainSize, expectedTestSize, numAttributes, new Random(0));

					if (split.get(0).size() != expectedTrainSize) {
						throw new IllegalStateException("Expecting train size " + expectedTrainSize + " but was " + split.get(0).size());
					}
					if (split.get(1).size() != expectedTestSize) {
						throw new IllegalStateException("Expecting test size " + expectedTestSize + " but was " + split.get(1).size());
					}

					for (int i = 0; i < 2; i++) {
						if (split.get(i).getNumAttributes() != numAttributes) {
							throw new IllegalStateException("Expecting number of attributes " + numAttributes + " but was " + split.get(i).getNumAttributes());
						}
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}
}
