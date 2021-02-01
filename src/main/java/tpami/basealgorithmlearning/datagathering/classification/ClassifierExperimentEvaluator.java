package tpami.basealgorithmlearning.datagathering.classification;

import org.api4.java.ai.ml.classification.IClassifier;
import org.api4.java.algorithm.Timeout;

import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import tpami.basealgorithmlearning.IConfigContainer;
import tpami.basealgorithmlearning.datagathering.ALearnerExperimentEvaluator;
import weka.classifiers.AbstractClassifier;

public class ClassifierExperimentEvaluator extends ALearnerExperimentEvaluator {

	private final String classifierName;

	public ClassifierExperimentEvaluator(final IConfigContainer container, final String classifierName, final Timeout to) throws ClassNotFoundException {
		super(container, to);
		this.classifierName = classifierName;
	}

	@Override
	public String getNameOfEvaluatedAlgorithm() {
		return this.classifierName;
	}

	@Override
	public IClassifier getClassifier(final String optionString) throws Exception {
		return new WekaClassifier(AbstractClassifier.forName(Class.forName(this.classifierName).getName(), optionString.split(" ")));
	}

	@Override
	public String getBeforeMFSuffix() {
		return null;
	}

}
