package tpami.safeguard;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.junit.BeforeClass;
import org.junit.Test;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.safeguard.impl.BaseComponentEvaluationTimePredictor;
import tpami.safeguard.impl.MetaFeatureContainer;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;
import weka.core.Instances;

public class BasicComponentPredictorTest {
	private static final ISimpleHierarchicalRFSafeGuardConfig CONFIG = ConfigFactory.create(ISimpleHierarchicalRFSafeGuardConfig.class);

	private static final String DEFAULT_FILE_PATTERN = "runtimes_%s_default.csv";
	private static final String PARAM_FILE_PATTERN = "runtimes_%s_parametrized.csv";

	protected static IComponentRepository cl;
	private static Map<String, String> learners = new HashMap<>();
	private static MetaFeatureContainer metaFeaturesTrain;

	@BeforeClass
	public static void setup() throws IOException {
		Arrays.stream(DataBasedComponentPredictorUtil.WEKA_CLASSES).forEach(x -> {
			learners.put(x.getSimpleName().toLowerCase(), x.getName());
		});
		cl = new ComponentSerialization().deserializeRepository(new File("testrsc/automl/searchmodels/weka/base/index.json"));
		Instances dataset = new Instances(new FileReader(new File("testrsc/car.arff")));
		dataset.setClassIndex(dataset.numAttributes() - 1);

		metaFeaturesTrain = new MetaFeatureContainer(new WekaInstances(dataset));
	}

	@Test
	public void testRandomLearners() throws Exception {
		for (Entry<String, String> learnerEntry : learners.entrySet()) {
			System.out.println(learnerEntry);

			KVStoreCollection paramData = this.getData(learnerEntry.getKey(), false);
			if (paramData != null) {
				Map<String, Set<String>> values = new HashMap<>();
				paramData.stream().forEach(x -> {
					x.entrySet().stream().forEach(y -> values.computeIfAbsent(y.getKey(), t -> new HashSet<>()).add(y.getValue() + ""));
				});
				values.remove("openmlid");
				values.remove("fitsize");
				Arrays.asList("applicationtime_def", "applicationsize", "totalsize", "seed", "applicationtime", "fittime", "algorithmoptions", "numattributes", "algorithm", "fittime_def").stream().forEach(values::remove);
				System.out.println(values);

				try {
					BaseComponentEvaluationTimePredictor pred = new BaseComponentEvaluationTimePredictor(learnerEntry.getKey(), this.getData(learnerEntry.getKey(), true), this.getData(learnerEntry.getKey(), false));

					Random rand = new Random();
					for (int i = 0; i < 10; i++) {
						ComponentInstance ci = ComponentUtil.getRandomParameterizationOfComponent(cl.getComponent(learnerEntry.getValue()), rand);
						System.out.println(pred.predictEvaluationTime(ci, metaFeaturesTrain, 1000));
					}
				} catch (Exception e) {
					System.out.println(ExceptionUtils.getStackTrace(e));
				}
			}
		}

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
