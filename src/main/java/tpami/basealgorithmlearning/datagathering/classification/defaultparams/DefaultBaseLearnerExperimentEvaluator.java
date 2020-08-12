package tpami.basealgorithmlearning.datagathering.classification.defaultparams;

import java.util.Collection;
import java.util.Map;

import org.api4.java.ai.ml.classification.IClassifier;
import org.api4.java.algorithm.Timeout;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import tpami.basealgorithmlearning.datagathering.ALearnerExperimentEvaluator;
import weka.classifiers.AbstractClassifier;

public class DefaultBaseLearnerExperimentEvaluator extends ALearnerExperimentEvaluator {

	private final String classifierName;

	public DefaultBaseLearnerExperimentEvaluator(final String configFileName, final String classifierName, final Timeout to) throws ClassNotFoundException {
		super(new DefaultBaseLearnerConfigContainer(configFileName, classifierName), to);
		this.classifierName = classifierName;
	}

	@Override
	public String getNameOfEvaluatedClassifier() {
		return this.classifierName;
	}

	@Override
	public IClassifier getClassifier() throws Exception {
		return new WekaClassifier(AbstractClassifier.forName(Class.forName(this.classifierName).getName(), null));
	}

	@Override
	public void checkFail(final Collection<ExperimentDBEntry> failedExperimentsOnThisDataset, final Map<String, String> experimentKeys) throws ExperimentFailurePredictionException {

		/* currently no fail check implemented */
	}

}
