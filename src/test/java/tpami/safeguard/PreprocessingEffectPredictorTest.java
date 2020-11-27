package tpami.safeguard;

import static org.junit.Assert.assertNotNull;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

import org.junit.Test;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.ComponentUtil;
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

		IComponent searchC = null;
		switch (split[0]) {
		case "bestfirst":
			searchC = cl.getComponent(BestFirst.class.getName());
			break;
		case "greedystepwise":
			searchC = cl.getComponent(GreedyStepwise.class.getName());
			break;
		case "ranker":
			searchC = cl.getComponent(Ranker.class.getName());
			break;
		}

		IComponent evalC = null;
		switch (split[1]) {
		case "cfssubseteval":
			evalC = cl.getComponent(CfsSubsetEval.class.getName());
			break;
		case "correlationattributeeval":
			evalC = cl.getComponent(CorrelationAttributeEval.class.getName());
			break;
		case "gainratioattributeeval":
			evalC = cl.getComponent(GainRatioAttributeEval.class.getName());
			break;
		case "infogainattributeeval":
			evalC = cl.getComponent(InfoGainAttributeEval.class.getName());
			break;
		case "onerattributeeval":
			evalC = cl.getComponent(OneRAttributeEval.class.getName());
			break;
		case "principalcomponents":
			evalC = cl.getComponent(PrincipalComponents.class.getName());
			break;
		case "relieffattributeeval":
			evalC = cl.getComponent(ReliefFAttributeEval.class.getName());
			break;
		case "symmetricaluncertattributeeval":
			evalC = cl.getComponent(SymmetricalUncertAttributeEval.class.getName());
			break;
		}

		assertNotNull(evalC);
		assertNotNull(searchC);

		IComponent asC = cl.getComponent(AttributeSelection.class.getName());

		ComponentInstance asCI = ComponentUtil.getDefaultParameterizationOfComponent(asC);
		ComponentInstance searchCI = ComponentUtil.getDefaultParameterizationOfComponent(searchC);
		ComponentInstance evalCI = ComponentUtil.getDefaultParameterizationOfComponent(evalC);

		asCI.getSatisfactionOfRequiredInterfaces().put("search", Arrays.asList(searchCI));
		asCI.getSatisfactionOfRequiredInterfaces().put("eval", Arrays.asList(evalCI));

		return asCI;
	}

}
