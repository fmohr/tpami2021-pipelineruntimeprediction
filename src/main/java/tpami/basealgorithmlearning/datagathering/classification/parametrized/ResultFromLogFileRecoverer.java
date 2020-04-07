package tpami.basealgorithmlearning.datagathering.classification.parametrized;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.db.IDatabaseAdapter;

public class ResultFromLogFileRecoverer {
	public static void main(final String[] args) throws Exception {
		if (args.length != 5) {
			System.err.println("Use 5 arguments: \n\t1) class of classifier\n\t2) db config file\n\t3) folder of logs\n\t4) prefix \n\t5) suffix");
		}
		Class c = Class.forName(args[0]);
		BaseLearnerConfigContainer cont = new BaseLearnerConfigContainer(args[1], c.getName());
		IDatabaseAdapter adapter = cont.getAdapter();

		File folder = new File(args[2]);
		String prefix = args[3];
		String suffix = args[4];

		String table = "evaluations_classifiers_" + c.getSimpleName().toLowerCase() + "_configured";

		int lines = 0;
		try (Stream<Path> walk = Files.walk(folder.toPath())) {
			List<String> result = walk.filter(p -> Files.isRegularFile(p) && p.toFile().getName().startsWith(prefix) && p.toFile().getName().endsWith(suffix)).map(x -> x.toString()).collect(Collectors.toList());

			for (String file : result) {
				int updates = 0;
				try (BufferedReader br = new BufferedReader(new FileReader(new File(file)))) {
					String line = null;
					while ((line = br.readLine()) != null) {
						lines ++;
						if (line.contains("Updating")) {
							String mapString = line.substring(line.indexOf("following map: ") + "following map: ".length());
							Map<String, Object> map = explodeMap(mapString);
							int id = Integer.parseInt(line.substring(100).split(" ")[6]);
							Map<String, String> cond = new HashMap<>();
							cond.put("experiment_id", "" + id);
							IKVStore row = adapter.getRowsOfTable(table, cond).iterator().next();
							if (row.get("train_start") == null) {
								System.out.println("Updating " + id);
								System.out.println(row.get("train_start"));
								map.put("time_started", map.get("train_start"));
								map.put("time_end", map.get("test_end"));
								adapter.update(table, map, cond);
								updates ++;
							}
							else {
								System.out.println(id + " is already there: " + row.get("train_start"));
							}
						}
						if (lines % 10000 == 0) {
							System.out.println("Passed " + lines + " lines.");
						}
					}
				}
				System.out.println("Conducted " + updates + " updates on file " + file);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static Map<String, Object> explodeMap(final String s) throws IOException {
		Map<String, Object> myMap = new HashMap<>();
		String t = s;
		t = t.substring(1);
		t = t.substring(0, t.length() - 1);
		//		StringBuilder sb = new StringBuilder();
		boolean inList = false;
		StringBuilder curVal = new StringBuilder();

		for (int i = 0; i < t.length(); i++) {
			char c =t.charAt(i);
			if (c == '[') {
				inList = true;
			}
			if (c == ']') {
				inList = false;
			}
			if (c == ',' && !inList) {
				//				System.out.println(curVal);
				String[] parts = curVal.toString().trim().split("=");
				myMap.put(parts[0], parts[1]);
				curVal.setLength(0);
			}
			else {
				curVal.append(c);
			}
		}
		String[] parts = curVal.toString().trim().split("=");
		myMap.put(parts[0], parts[1]);
		return myMap;
	}
}
