package tpami.safeguard;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.Random;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.mlplan.multiclass.wekamlplan.weka.WekaPipelineFactory;
import tpami.safeguard.impl.MetaFeatureContainer;
import tpami.safeguard.util.MLComponentInstanceWrapper;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class SafeGuardPredictionTest {
	private static final File SEARCH_SPACE_CONFIG_FILE = new ResourceFile("automl/searchmodels/weka/weka-all-autoweka.json");
	private static final File DEFAULT_COMPONENTS_DATA = new File("python/data/runtimes_all_default_nooutliers.csv");

	private static final String[] META_LEARNERS = { Bagging.class.getName() };

	private static ComponentLoader cl;
	private static SimpleHierarchicalRFSafeGuard safeGuard;

	@BeforeClass
	public static void setup() throws Exception {
		cl = new ComponentLoader(SEARCH_SPACE_CONFIG_FILE);
		long startTime = System.currentTimeMillis();
		System.out.println("Instantiate safe guard...");
//		safeGuard = new SimpleHierarchicalRFSafeGuard(DEFAULT_COMPONENTS_DATA, 1000, 1002, 1018, 1019, 1020);
		System.out.println("Building safe guard took " + (System.currentTimeMillis() - startTime) + "ms");
	}

	@Test
	public void testNothing() {
		assertTrue("", true);
	}

	@Ignore
	@Test
	public void testDefaultConfigurationPipeline() throws Exception {
		// Classifier
		ComponentInstance metaLearner = ComponentUtil.defaultParameterizationOfComponent(cl.getComponentWithName(AdaBoostM1.class.getName()));
		ComponentInstance baseLearner = ComponentUtil.defaultParameterizationOfComponent(cl.getComponentWithName(J48.class.getName()));
		metaLearner.getSatisfactionOfRequiredInterfaces().put("W", baseLearner);

		// Preprocessor
		ComponentInstance preprocessor = ComponentUtil.defaultParameterizationOfComponent(cl.getComponentWithName(AttributeSelection.class.getName()));
		ComponentInstance searcher = ComponentUtil.defaultParameterizationOfComponent(cl.getComponentWithName(BestFirst.class.getName()));
		ComponentInstance evaluator = ComponentUtil.defaultParameterizationOfComponent(cl.getComponentWithName(CfsSubsetEval.class.getName()));
		preprocessor.getSatisfactionOfRequiredInterfaces().put("search", searcher);
		preprocessor.getSatisfactionOfRequiredInterfaces().put("eval", evaluator);

		// Pipeline
		ComponentInstance pipeline = ComponentUtil.defaultParameterizationOfComponent(cl.getComponentWithName("pipeline"));
		pipeline.getSatisfactionOfRequiredInterfaces().put("preprocessor", preprocessor);
		pipeline.getSatisfactionOfRequiredInterfaces().put("classifier", metaLearner);

		ILabeledDataset<?> dataset = ArffDatasetAdapter.readDataset(new File("car.arff"));
		List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, 0, .7);

		double predictedRuntime = safeGuard.predictEvaluationTime(pipeline, split.get(0), split.get(1));
		System.out.println("Predicted: " + predictedRuntime + "s");

		WekaPipelineFactory factory = new WekaPipelineFactory();
		IWekaClassifier classifier = factory.getComponentInstantiation(pipeline);

		SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
		long startTime = System.currentTimeMillis();
		ILearnerRunReport report = executor.execute(classifier, split.get(0), split.get(1));
		double errorRate = EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList());
		System.out.println(errorRate);

		System.out.println("Actual: " + ((double) (System.currentTimeMillis() - startTime) / 1000) + "s");
	}

	@Test
	public void testPlainPipeline() throws Exception {
		Instances dataset = new Instances(new FileReader(new File("car.arff")));
		dataset.setClassIndex(dataset.numAttributes() - 1);
		List<IWekaInstances> split = WekaUtil.getStratifiedSplit(new WekaInstances(dataset), new Random(0), 0.7);

		AdaBoostM1 ada = new AdaBoostM1();
		ada.setClassifier(new J48());

		AttributeSelection as = new AttributeSelection();
		as.setSearch(new BestFirst());
		as.setEvaluator(new CfsSubsetEval());

		long startTime = System.currentTimeMillis();

		as.SelectAttributes(split.get(0).getInstances());
		Instances ppdata = as.reduceDimensionality(split.get(0).getInstances());

		ada.buildClassifier(ppdata);

		Instances pptestData = as.reduceDimensionality(split.get(1).getInstances());
		new Evaluation(ppdata).evaluateModel(ada, pptestData);
		long stopTime = System.currentTimeMillis();

		System.out.println((((double) (stopTime - startTime)) / 1000) + "s");
	}

	@Ignore
	@Test
	public void testDefaultConfigBaselearnerPrediction() throws Exception {
		ComponentInstance baselearner = this.sampleBaselearner(11);
		MetaFeatureContainer mf = new MetaFeatureContainer(100, 10);
		System.out.println(safeGuard.predictInductionTime(new MLComponentInstanceWrapper(baselearner), mf));
	}

	@Ignore
	@Test
	public void testDefaultConfigPreprocessorPrediction() throws Exception {
		ComponentInstance preprocessor = this.samplePreprocessor(0);
		MetaFeatureContainer mf = new MetaFeatureContainer(100, 10);
		System.out.println(preprocessor);
		System.out.println("Preprocessor: " + safeGuard.predictInductionTime(new MLComponentInstanceWrapper(preprocessor), mf));
	}

	private ComponentInstance samplePipeline() {
		return null;
	}

	private ComponentInstance sampleBaselearner(final long seed) {
		return this.sampleComponentInstance("BaseClassifier", seed);
	}

	private ComponentInstance samplePreprocessor(final long seed) {
		return this.sampleComponentInstance("AbstractPreprocessor", seed);
	}

	private ComponentInstance sampleComponentInstance(final String requiredInterface, final long seed) {
		List<ComponentInstance> components = (List<ComponentInstance>) ComponentUtil.getAllAlgorithmSelectionInstances(requiredInterface, cl.getComponents());
		return components.get(new Random(seed).nextInt(components.size()));
	}

}
