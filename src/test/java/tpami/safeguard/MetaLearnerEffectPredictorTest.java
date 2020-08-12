package tpami.safeguard;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.aeonbits.owner.ConfigFactory;
import org.junit.BeforeClass;
import org.junit.Test;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.jaicore.components.serialization.ComponentLoader;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.safeguard.api.EMetaFeature;
import tpami.safeguard.impl.BaseComponentEvaluationTimePredictor;
import tpami.safeguard.impl.MetaFeatureContainer;
import tpami.safeguard.impl.MetaLearnerEvaluationTimePredictor;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import tpami.safeguard.util.MLComponentInstanceWrapper;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MetaLearnerEffectPredictorTest {

	private static final ISimpleHierarchicalRFSafeGuardConfig CONFIG = ConfigFactory.create(ISimpleHierarchicalRFSafeGuardConfig.class);

	private static ComponentLoader cl;

	@BeforeClass
	public static void setup() throws IOException {
		cl = new ComponentLoader(new ResourceFile("automl/searchmodels/weka/weka-all-autoweka.json"));
	}

	@Test
	public void testMetalearnerEffectModel() throws Exception {
		BaseComponentEvaluationTimePredictor basePred = new BaseComponentEvaluationTimePredictor("j48", this.getData("j48", true), this.getData("j48", false));
		Instances dataset = new Instances(new FileReader(new File("car.arff")));
		dataset.setClassIndex(dataset.numAttributes() - 1);
		MetaFeatureContainer mf = new MetaFeatureContainer(new WekaInstances(dataset));
		System.out.println(Double.MAX_VALUE);

		for (String name : CONFIG.getMetaLearnerTransformEffect()) {
			System.out.println(name);
			File csvFile = new File(CONFIG.getMetaLearnerTransformEffectDirectory(), String.format(ISimpleHierarchicalRFSafeGuardConfig.FILE_PATTERN_METALEARNER, name));
			System.out.println(csvFile);
			KVStoreCollection metaLearnerEffectData = DataBasedComponentPredictorUtil.readCSV(csvFile, new HashMap<>());

			Map<String, Set<String>> values = new HashMap<>();
			metaLearnerEffectData.stream().forEach(x -> x.entrySet().stream().forEach(y -> values.computeIfAbsent(y.getKey(), t -> new HashSet<>()).add(y.getValue() + "")));
			values.entrySet().stream().forEach(System.out::println);

			MetaLearnerEvaluationTimePredictor pred = new MetaLearnerEvaluationTimePredictor(name, metaLearnerEffectData);

			for (int i = 0; i < 10; i++) {
				System.out.println(pred.predictEvaluationTime(new MLComponentInstanceWrapper(this.getInstance(name, new Random())), basePred, mf, new MetaFeatureContainer(1000, mf.getFeature(EMetaFeature.NUM_ATTRIBUTES))));
			}
		}
	}

	private ComponentInstance getInstance(final String name, final Random rand) {
		ComponentInstance bci = ComponentUtil.getRandomParameterizationOfComponent(cl.getComponentWithName(J48.class.getName()), rand);

		ComponentInstance ci = null;
		switch (name) {
		case "adaboostm1":
			ci = ComponentUtil.getRandomParameterizationOfComponent(cl.getComponentWithName(AdaBoostM1.class.getName()), rand);
			break;
		case "bagging":
			ci = ComponentUtil.getRandomParameterizationOfComponent(cl.getComponentWithName(Bagging.class.getName()), rand);
			break;
		case "logitboost":
			ci = ComponentUtil.getRandomParameterizationOfComponent(cl.getComponentWithName(LogitBoost.class.getName()), rand);
			break;
		case "randomcommittee":
			ci = ComponentUtil.getRandomParameterizationOfComponent(cl.getComponentWithName(RandomCommittee.class.getName()), rand);
			break;
		case "randomsubspace":
			ci = ComponentUtil.getRandomParameterizationOfComponent(cl.getComponentWithName(RandomSubSpace.class.getName()), rand);
			break;
		}
		ci.getSatisfactionOfRequiredInterfaces().put("W", bci);
		return ci;
	}

	private KVStoreCollection getData(final String learner, final boolean defaultParams) throws IOException {
		File file = null;
		if (defaultParams) {
			file = new File(CONFIG.getBasicComponentsForDefaultRuntimeDirectory(), String.format(ISimpleHierarchicalRFSafeGuardConfig.FILE_PATTERN_BASIC_DEF, learner));
		} else {
			file = new File(CONFIG.getBasicComponentsForDefaultRuntimeDirectory(), String.format(ISimpleHierarchicalRFSafeGuardConfig.FILE_PATTERN_BASIC_PAR, learner));
		}
		try {
			return DataBasedComponentPredictorUtil.readCSV(file, new HashMap<>());
		} catch (IllegalArgumentException e) {
			return null;
		}
	}

}
