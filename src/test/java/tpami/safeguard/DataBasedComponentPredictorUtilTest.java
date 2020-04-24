package tpami.safeguard;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import weka.core.Instances;

public class DataBasedComponentPredictorUtilTest {

	private static final File DEFAULT_CSV_FILE = new File("python/data/runtimes_all_default_nooutliers.csv");
	private static final int EXPECTED_ROWS_RUNTIMES_ALL_DEFAULT = 1017627;
	private static final List<String> EXPECTED_HEADERS = Arrays.asList("openmlid", "totalsize", "fitsize", "applicationsize", "fitattributes", "seed", "algorithm", "fittime", "applicationtime");

	/**
	 * Test whether we can read in the csv file correctly.
	 * @throws IOException
	 */
	@Test
	public void testReadingCSVFile() throws IOException {
		KVStoreCollection col = DataBasedComponentPredictorUtil.readCSV(DEFAULT_CSV_FILE, new HashMap<>());
		assertEquals("Number of instances is not correct", EXPECTED_ROWS_RUNTIMES_ALL_DEFAULT, col.size());
		assertEquals("Number of headers deviates from the expected number", EXPECTED_HEADERS.size(), col.get(0).keySet().size());
		assertTrue("Not all expected headers are included", EXPECTED_HEADERS.containsAll(col.get(0).keySet()));
	}

	/**
	 * Test whether we can read in the csv file correctly.
	 * @throws Exception
	 */
	@Test
	public void testKVStoreCollectionToInstances() throws Exception {
		KVStoreCollection col = DataBasedComponentPredictorUtil.readCSV(DEFAULT_CSV_FILE, new HashMap<>());
		col.setCollectionID(DEFAULT_CSV_FILE.getName());
		Instances data = DataBasedComponentPredictorUtil.kvStoreCollectionToWekaInstances(col, "fittime", "fitsize", "fitattributes");

		assertEquals("Number of attributes deviates from the expected number", 3, data.numAttributes());
		assertEquals("Number of instances deviates from the expected number", col.size(), data.size());
		for (int i = 0; i < data.numAttributes(); i++) {
			assertTrue("Failed to represent attribute " + data.attribute(i).name() + " by a numeric attribute", data.attribute(i).isNumeric());
		}
	}

}
