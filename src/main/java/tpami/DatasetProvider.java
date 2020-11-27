package tpami;

import java.io.File;
import java.util.List;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import tpami.basealgorithmlearning.datagathering.ExperimentUtil;
import tpami.basealgorithmlearning.datagathering.classification.defaultparams.IDefaultBaseLearnerExperimentConfig;

public class DatasetProvider {

	public static void main(final String[] args) {
		List<Integer> ids = ((IDefaultBaseLearnerExperimentConfig)ConfigFactory.create(IDefaultBaseLearnerExperimentConfig.class).loadPropertiesFromFile(new File("conf/experiments/defaultparams/baselearner.conf"))).openMLIDs();
		ExperimentUtil util = new ExperimentUtil();
		util.setLoggerName(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);
		int maxRequiredInstances = 100000 + 1500; // maximum training with 100k instances and max predictions with 1500 instances.

		if  (args.length > 0) {
			final int index = Integer.parseInt(args[0]);
			final int id = ids.get(index);
			ids.removeIf(i -> i != id);
		}

		System.out.println("Iterating over ids: " + ids);
		ids.forEach(id -> {
			try {
				System.out.println("Readining in " + id);
				ILabeledDataset<?> ds = OpenMLDatasetReader.deserializeDataset(id);
				System.out.println("Done");
				int maxInstances = Math.min(maxRequiredInstances, ExperimentUtil.getMaximumInstancesManagableForDataset(ds));
				System.out.println("Max instances on this dataset is  " + maxInstances);
				if (maxInstances > ds.size()) {
					int extensionSize = maxInstances - ds.size();

					/* get extension dataset if available */
					if (ExperimentUtil.doesExtensionExist(id) && util.getSMOTEDatasetExtension(id).size() >= maxInstances - ds.size()) {
						System.out.println("Extension is already built and big enough.");
					}
					else {
						System.out.println("Create extension with SMOTE of size " + extensionSize);
						util.createSMOTEExtensionOfOpenMLDataset(id, extensionSize);
						System.out.println("Done");
					}
				}
				else {
					System.out.println("The dataset is big enough to cover all relevant experiments. No extension is needed.");
				}

			} catch (Exception e) {
				e.printStackTrace();
			}
		});
	}
}
