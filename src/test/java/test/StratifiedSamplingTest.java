package test;

import java.io.File;
import java.util.List;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;

import ai.libs.jaicore.basic.MathExt;
import ai.libs.jaicore.basic.Tester;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import tpami.basealgorithmlearning.datagathering.ExperimentUtil;
import tpami.basealgorithmlearning.datagathering.classification.defaultparams.IDefaultBaseLearnerExperimentConfig;

public class StratifiedSamplingTest extends Tester {

	@Rule
	public Timeout globalTimeout = Timeout.seconds(1800); // this test may run 20 minutes

	@Test
	public void testSplittability() throws DatasetDeserializationFailedException, SplitFailedException, InterruptedException {
		List<Integer> ids = ((IDefaultBaseLearnerExperimentConfig)ConfigFactory.create(IDefaultBaseLearnerExperimentConfig.class).loadPropertiesFromFile(new File("conf/experiments/defaultparams/baselearner.conf"))).openMLIDs();
		ExperimentUtil util = new ExperimentUtil();
		util.setLoggerName(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);

		int k = 0;
		for (int id : ids) {
			k ++;
			this.logger.info("Now splitting id {}. Progress: {}%", id, MathExt.round(k * 100.0 / ids.size(), 2));
			ILabeledDataset<?> ds = OpenMLDatasetReader.deserializeDataset(id);
			SplitterUtil.getLabelStratifiedTrainTestSplit(ds, 0, 0.7);
		}
	}

}
