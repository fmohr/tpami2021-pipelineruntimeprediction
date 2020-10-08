package tpami.pipelinemeasurement.parametrized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.algorithm.Timeout;
import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.jaicore.basic.reconstruction.ReconstructionPlan;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import tpami.basealgorithmlearning.IConfigContainer;
import tpami.basealgorithmlearning.datagathering.ExperimentUtil;
import tpami.basealgorithmlearning.regression.BasicDatasetFeatureGenerator;
import tpami.basealgorithmlearning.regression.DatasetVarianceFeatureGenerator;

public class PipelineExperimentEvaluator implements IExperimentSetEvaluator, ILoggingCustomizable {

	/* meta feature generators */
	private static final BasicDatasetFeatureGenerator MGENERATOR_BASIC = new BasicDatasetFeatureGenerator();
	private static final DatasetVarianceFeatureGenerator MGENERATOR_VARIANCE = new DatasetVarianceFeatureGenerator();

	private Logger logger = LoggerFactory.getLogger(PipelineExperimentEvaluator.class);
	private final IConfigContainer container;
	private final Timeout to;

	public PipelineExperimentEvaluator(final IConfigContainer container, final Timeout to) {
		super();
		this.container = container;
		this.to = to;
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		try {
			ObjectMapper om = new ObjectMapper();
			JsonNode setup = om.readTree(experimentEntry.getExperiment().getValuesOfKeyFields().get("setup"));

			int openmlid = setup.get("openmlid").asInt();
			int numinstances = setup.get("numinstances").asInt();
			int numattributes = setup.get("numattributes").asInt();
			WekaClassifier pipeline = (WekaClassifier)om.readValue(setup.get("pipeline").toString(), ReconstructionPlan.class).reconstructObject();

			/* load data set and train and predict */
			List<ILabeledDataset<?>> split = new ExperimentUtil().createSizeAdjustedTrainTestSplit(openmlid, numinstances, 1000, numattributes, new Random(0));

			/* compute meta-features */
			Map<String, Object> metaFeatures = new HashMap<>();
			MGENERATOR_BASIC.setSuffix("");
			MGENERATOR_VARIANCE.setSuffix("");
			metaFeatures.putAll(MGENERATOR_BASIC.getFeatureRepresentation(split.get(0)));
			metaFeatures.putAll(MGENERATOR_VARIANCE.getFeatureRepresentation(split.get(0)));
			List<String> relevantFeatures = Arrays.asList("totalvariance", "numberofcategories", "numericattributesafterbinarization", "numattributes", "numsymbolicattributes", "attributestocover90pctvariance", "numinstances",
					"attributestocover99pctvariance", "attributestocover50pctvariance", "numlabels", "numnumericattributes", "attributestocover95pctvariance");
			for (String key : new ArrayList<>(metaFeatures.keySet())) {
				if (!relevantFeatures.contains(key)) {
					metaFeatures.remove(key);
				}
			}
			processor.processResults(metaFeatures);

			long fitStart = System.currentTimeMillis();
			pipeline.fit(split.get(0));
			long fitStop = System.currentTimeMillis();
			pipeline.predict(split.get(1));
			long predictStop = System.currentTimeMillis();

			/* compute results */
			Map<String, Object> results = new HashMap<>();
			results.put("traintimeinms", (fitStop - fitStart));
			results.put("timeforpredictionsinms", (predictStop - fitStop));
			results.put("predictedinstances", split.get(1).size());
			processor.processResults(results);
		}
		catch (Exception e) {
			throw new ExperimentEvaluationFailedException(e);
		}

	}

	public IConfigContainer getContainer() {
		return this.container;
	}

}
