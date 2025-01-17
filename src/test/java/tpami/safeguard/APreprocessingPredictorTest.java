package tpami.safeguard;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import org.aeonbits.owner.ConfigFactory;
import org.junit.BeforeClass;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import tpami.safeguard.util.DataBasedComponentPredictorUtil;

public abstract class APreprocessingPredictorTest {

	private static final ISimpleHierarchicalRFSafeGuardConfig CONFIG = ConfigFactory.create(ISimpleHierarchicalRFSafeGuardConfig.class);
	protected static final List<String> AVAILABLE_PREPROCESSORS = CONFIG.getPreprocessorsForTransformEffect();
	protected static final int INDEX = 0;

	protected static ComponentSerialization serializer = new ComponentSerialization();
	protected static IComponentRepository cl;

	@BeforeClass
	public static void setup() throws IOException {
		cl = serializer.deserializeRepository(new File("testrsc/automl/searchmodels/weka/base/index.json"));
	}

	protected static KVStoreCollection getData(final String preprocessor) throws IOException {
		return DataBasedComponentPredictorUtil.readCSV(getDatasetFile(preprocessor), new HashMap<>());
	}

	protected static File getDatasetFile(final String preprocessor) {
		return new File(CONFIG.getPreprocessorsForTransformEffectDirectory(), String.format(ISimpleHierarchicalRFSafeGuardConfig.FILE_PATTERN_PREPROCESSOR, preprocessor));
	}
}
