package tpami.safeguard;

import ai.libs.hasco.model.ComponentInstance;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class RandomForestMetaLearnerPredictor implements IMetaLearnerPredictor {

	private Instances schema;
	private Classifier numBaseLearnerBuilds = new RandomForest();
	private Classifier numInstances = new RandomForest();
	private Classifier numAttributes = new RandomForest();
	private Classifier inductionNumBaseLearnerInferences = new RandomForest();
	private Classifier inferenceNumBaseLearnerInferences = new RandomForest();

	public RandomForestMetaLearnerPredictor() {

	}

	@Override
	public double predictInductionTime(final ComponentInstance ciw, final IComponentPredictor iComponentPredictor, final double[] metaFeaturesTrain) {
		return 0;
	}

	@Override
	public double predictInferenceTime(final MLComponentInstanceWrapper ciw, final IComponentPredictor iComponentPredictor, final double[] metaFeaturesTest) {
		// TODO Auto-generated method stub
		return 0;
	}

	private double predictNumBaselearnerInductions() {
		return this.numBaseLearnerBuilds.classifyInstance(instance);
	}

	private double[] predictSubMetaFeatures(final double[] metaFeatures) {
		// TODO Auto-generated method stub
		return null;
	}

	private double predictNumBaseLearnInferencesDuringInduction() {
		// TODO Auto-generated method stub
		return 0;
	}

	private double predictNumBaseLearnerInferencesDuringInference() {
		// TODO Auto-generated method stub
		return 0;
	}

	private Classifier getModel() {
		RandomForest forest = new RandomForest();
		forest.setNumIterations(100);
		return forest;
	}

}
