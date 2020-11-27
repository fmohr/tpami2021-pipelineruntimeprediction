package tpami.internalvalidation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import tpami.safeguard.SimpleHierarchicalRFSafeGuard;

public class InternalValidator {

	private static final File SPLIT_FOLDER = new File("internal-val/");

	private static final boolean TEST_BASE_DEF = true;
	private static final boolean TEST_BASE_PRM = true;

	public InternalValidator(final List<String> testIDs) throws Exception {
		new ComponentSerialization().deserializeRepository(new ResourceFile("automl/searchmodels/weka/weka-all-autoweka.jso"));
		SimpleHierarchicalRFSafeGuard guard = new SimpleHierarchicalRFSafeGuard(testIDs.stream().mapToInt(x -> Integer.parseInt(x)).toArray(), null, null, null);

		if (TEST_BASE_DEF) {
			this.testDefaultBaseComponents(testIDs);
		}
		if (TEST_BASE_PRM) {
			this.testParameterizedBaseComponents(testIDs);
		}

	}

	private void testDefaultBaseComponents(final List<String> testIDs) {

	}

	private void testParameterizedBaseComponents(final List<String> testIDs) {

	}

	public static void main(final String[] args) throws IOException {
		SPLIT_FOLDER.mkdirs();

		int folds = 5;
		List<String> ids = FileUtil.readFileAsList(new File("openmlids.txt"));
		List<List<String>> idFolds = new ArrayList<>(5);
		IntStream.range(0, folds).forEach(i -> idFolds.add(new ArrayList<>()));

		for (int i = 0; i < ids.size(); i++) {
			idFolds.get(i % folds).add(ids.get(i));
		}

		for (int fold = 0; fold < folds; fold++) {
			List<String> train = new ArrayList<>();
			List<String> test = new ArrayList<>();

			for (int f = 0; f < folds; f++) {
				if (f == fold) {
					test.addAll(idFolds.get(f));
				} else {
					train.addAll(idFolds.get(fold));
				}
			}

			writeFile(new File(SPLIT_FOLDER, "openmlids-" + folds + "cv-" + fold + "-train.txt"), train);
			writeFile(new File(SPLIT_FOLDER, "openmlids-" + folds + "cv-" + fold + "-test.txt"), test);
		}

	}

	private static void writeFile(final File file, final List<String> ids) throws IOException {
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
			ids.stream().map(x -> x + "\n").forEach(x -> {
				try {
					bw.write(x);
				} catch (IOException e) {
					e.printStackTrace();
				}
			});
		}
	}

}
