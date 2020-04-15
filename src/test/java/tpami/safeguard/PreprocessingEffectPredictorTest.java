package tpami.safeguard;

import static org.junit.Assert.assertNotNull;

import java.io.FileReader;
import java.util.Random;

import org.junit.Test;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.safeguard.impl.MetaFeatureContainer;
import tpami.safeguard.impl.PreprocessingEffectPredictor;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Instances;

public class PreprocessingEffectPredictorTest extends APreprocessingPredictorTest {

	@Test
	public void testInstantiationOfPreprocessingPredictor() throws Exception {
		for (int i = 0; i < AVAILABLE_PREPROCESSORS.size(); i++) {
			String preprocessor = AVAILABLE_PREPROCESSORS.get(i);
			System.out.println(preprocessor);
			KVStoreCollection data = getData(preprocessor);

			Instances dataset = new Instances(new FileReader("car.arff"));
			dataset.setClassIndex(dataset.numAttributes() - 1);

			PreprocessingEffectPredictor pred = new PreprocessingEffectPredictor(preprocessor, data);

			MetaFeatureContainer prediction = pred.predictTransformedMetaFeatures(getInstance(preprocessor), new MetaFeatureContainer(new WekaInstances(dataset)));
			System.out.println(prediction);
		}
	}

	@Test
	public void testRandomParameterizationInstantiationOfPreprocessingPredictor() throws Exception {
		for (int i = 0; i < AVAILABLE_PREPROCESSORS.size(); i++) {
			String preprocessor = AVAILABLE_PREPROCESSORS.get(i);
			System.out.println(preprocessor);
			KVStoreCollection data = getData(preprocessor);

			Instances dataset = new Instances(new FileReader("car.arff"));
			dataset.setClassIndex(dataset.numAttributes() - 1);

			PreprocessingEffectPredictor pred = new PreprocessingEffectPredictor(preprocessor, data);

			MetaFeatureContainer prediction = pred.predictTransformedMetaFeatures(ComponentUtil.getRandomParametrization(getInstance(preprocessor), new Random()), new MetaFeatureContainer(new WekaInstances(dataset)));
			System.out.println(prediction);
		}
	}

	private static ComponentInstance getInstance(final String preprocessor) {
		String[] split = preprocessor.split("\\_");

		AttributeSelection as = new AttributeSelection();

		Component searchC = null;
		switch (split[0]) {
		case "bestfirst":
			searchC = cl.getComponentWithName(BestFirst.class.getName());
			break;
		case "greedystepwise":
			searchC = cl.getComponentWithName(GreedyStepwise.class.getName());
			break;
		case "ranker":
			searchC = cl.getComponentWithName(Ranker.class.getName());
			break;
		}

		Component evalC = null;
		switch (split[1]) {
		case "cfssubseteval":
			evalC = cl.getComponentWithName(CfsSubsetEval.class.getName());
			break;
		case "correlationattributeeval":
			evalC = cl.getComponentWithName(CorrelationAttributeEval.class.getName());
			break;
		case "gainratioattributeeval":
			evalC = cl.getComponentWithName(GainRatioAttributeEval.class.getName());
			break;
		case "infogainattributeeval":
			evalC = cl.getComponentWithName(InfoGainAttributeEval.class.getName());
			break;
		case "onerattributeeval":
			evalC = cl.getComponentWithName(OneRAttributeEval.class.getName());
			break;
		case "principalcomponents":
			evalC = cl.getComponentWithName(PrincipalComponents.class.getName());
			break;
		case "relieffattributeeval":
			evalC = cl.getComponentWithName(ReliefFAttributeEval.class.getName());
			break;
		case "symmetricaluncertattributeeval":
			evalC = cl.getComponentWithName(SymmetricalUncertAttributeEval.class.getName());
			break;
		}

		assertNotNull(evalC);
		assertNotNull(searchC);

		Component asC = cl.getComponentWithName(AttributeSelection.class.getName());

		ComponentInstance asCI = ComponentUtil.getDefaultParameterizationOfComponent(asC);
		ComponentInstance searchCI = ComponentUtil.getDefaultParameterizationOfComponent(searchC);
		ComponentInstance evalCI = ComponentUtil.getDefaultParameterizationOfComponent(evalC);

		asCI.getSatisfactionOfRequiredInterfaces().put("search", searchCI);
		asCI.getSatisfactionOfRequiredInterfaces().put("eval", evalCI);

		return asCI;
	}

}
