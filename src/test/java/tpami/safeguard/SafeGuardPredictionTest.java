package tpami.safeguard;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.List;
import java.util.Random;

import org.junit.BeforeClass;
import org.junit.Ignore;
import org.junit.Test;

import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.ResourceFile;
import tpami.safeguard.impl.MetaFeatureContainer;
import tpami.safeguard.util.MLComponentInstanceWrapper;
import weka.classifiers.meta.Bagging;

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
		safeGuard = new SimpleHierarchicalRFSafeGuard(DEFAULT_COMPONENTS_DATA, 1000, 1002, 1018, 1019, 1020);
		System.out.println("Building safe guard took " + (System.currentTimeMillis() - startTime) + "ms");
	}

	@Test
	public void testNothing() {
		assertTrue("", true);
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
