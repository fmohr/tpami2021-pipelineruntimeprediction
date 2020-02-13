package tpami.tablepreparer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.db.DatabaseUtil;
import ai.libs.jaicore.db.sql.SQLAdapter;

public class PipelineClassifierExtractor {
	public static void main(final String[] args) throws Exception {
		//		TableUtil.extractAndWriteUsedClassifiers("mlplanmlj2019reeval_aggregate");


		SQLAdapter ADAPTER = new SQLAdapter("localhost", "test", "test", "test", false);

		Map<String, Pair<Class<?>, Function<IKVStore, Object>>> transformations = new HashMap<>();
		transformations.put("p", new Pair<>(Double.class, r -> r.getAsDouble("b") * 0.5));
		transformations.put("q", new Pair<>(String.class, r -> r.getAsString("c").toLowerCase()));
		DatabaseUtil.createTableFromResult(ADAPTER, "SELECT * FROM test", Arrays.asList(), "newtable", Arrays.asList("p", "q"), transformations);
	}
}
