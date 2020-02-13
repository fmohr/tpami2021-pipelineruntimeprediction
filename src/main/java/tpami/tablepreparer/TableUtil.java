package tpami.tablepreparer;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.datastructure.kvstore.IKVStore;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import ai.libs.jaicore.basic.reconstruction.ReconstructionPlan;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.classification.pipeline.MLPipeline;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.classifiers.Classifier;

public class TableUtil {

	public static final SQLAdapter adapter = new SQLAdapter((IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("dbcon.conf")));

	public static void extractAndWriteUsedClassifiers(final String tablename) throws Exception {
		Iterator<IKVStore> it = adapter.getResultIteratorOfQuery("SELECT `eval_id`, `pipeline`, `pipelinealgorithms` FROM `" + tablename + "` WHERE pipelinealgorithms = ''", new ArrayList<>());
		ObjectMapper om = new ObjectMapper();
		while (it.hasNext()) {
			IKVStore row = it.next();
			if (row.get("pipelinealgorithms") == null || row.getAsString("pipelinealgorithms").isEmpty()) {
				Map<String, Object> update = new HashMap<>();
				ReconstructionPlan plan = om.readValue(row.getAsString("pipeline"), ReconstructionPlan.class);
				IWekaClassifier cl = (IWekaClassifier)plan.reconstructObject();
				ObjectNode node = om.createObjectNode();
				Classifier c = cl.getClassifier();
				if (c instanceof MLPipeline) {
					MLPipeline p = (MLPipeline)c;
					ASSearch searcher = p.getPreprocessors().get(0).getSearcher();
					ASEvaluation eval = p.getPreprocessors().get(0).getEvaluator();
					node.put("pp", searcher.getClass().getName() + "/" + eval.getClass().getName());
					node.put("c", p.getClassifier().getClass().getName());
				}
				else {
					node.put("pp", "null");
					node.put("c", c.getClass().getName());
				}
				update.put("pipelinealgorithms", om.writeValueAsString(node));
				Map<String, Object> cond = new HashMap<>();
				cond.put("eval_id", row.getAsInt("eval_id"));
				adapter.update(tablename, update, cond);
			}
		}
	}
}
