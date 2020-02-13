package tpami.basealgorithmlearning.regression;

import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.common.reconstruction.ReconstructionException;
import org.api4.java.datastructure.kvstore.IKVStore;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import ai.libs.jaicore.basic.reconstruction.ReconstructionPlan;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;
import ai.libs.jaicore.ml.core.dataset.DatasetUtil;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.converters.ArffSaver;

public class BaseAlgorithmDatasetPreparer {

	public static void main(final String[] arg) throws SQLException, JsonParseException, JsonMappingException, ReconstructionException, IOException, SplitFailedException, InterruptedException {
		SQLAdapter adapter = new SQLAdapter((IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("dbcon.conf")));
		final String table = "bl_logistic";

		/* create meta-database in form of a list of maps from database entries */
		List<IKVStore> rows = adapter.getRowsOfTable(table).stream().filter(r -> r.get("exception") == null).collect(Collectors.toList());
		System.out.println("Creating a dataset of " + rows.size() + " rows.");
		Set<String> keys = new HashSet<>();
		List<Map<String,Object>> metaDatasetAsMapList = new ArrayList<>();
		int i = 0;
		int n = rows.size();
		for (IKVStore row : rows) {
			//			analyzeDataDescr(row.getAsString("evaluationinputdata"));
			Map<String, Object> features = new HashMap<>();
			features.putAll(getDatasetFeatureRepresentation(row.getAsString("evaluationinputdata")));
			features.putAll(getClassifierFeatureRepresentation(row.getAsString("pipeline")));
			features.put("runtime", row.getAsDouble("traintime"));
			metaDatasetAsMapList.add(features);
			if (keys.isEmpty()) {
				keys.addAll(features.keySet());
			}
			else {
				if (!keys.equals(features.keySet())) {
					throw new IllegalStateException();
				}
			}
			i++;
			//			if (i > 100) {
			//				break;
			//			}
			System.out.println("Progress: " + i + "/" + n + " (" + Math.round(i * 100.0 / n) + "%)");
		}
		System.out.println("Features read completely. Now packing the data togther into a dataset.");

		/* now convert the list of maps into a dataset object */
		IWekaInstances metaDataset = new WekaInstances(DatasetUtil.getDatasetFromMapCollection(metaDatasetAsMapList, "runtime"));
		System.out.println("Done. Here is the data:");
		System.out.println(metaDataset);
		metaDataset.forEach(l -> System.out.println(l));

		/* write arff file with the values */
		ArffSaver saver = new ArffSaver();
		saver.setInstances(metaDataset.getInstances());
		saver.setFile(new File(table + ".arff"));
		saver.writeBatch();
	}

	private static WekaClassifier getWekaClassifier(final String description) throws JsonParseException, JsonMappingException, ReconstructionException, IOException {
		ReconstructionPlan plan = new ObjectMapper().readValue(description, ReconstructionPlan.class);
		return (WekaClassifier)plan.reconstructObject();
	}

	private static ILabeledDataset<?> getData(final String description) throws JsonParseException, JsonMappingException, ReconstructionException, IOException {
		String json = description.startsWith("{\"instructions\"") ? description : ( "{\"instructions\":" + description + "}");
		JsonNode node = new ObjectMapper().readTree(json);
		ArrayNode inst = (ArrayNode)node.get("instructions");
		if (inst.size() == 5) {
			inst.remove(0);
			inst.remove(0);
			inst.remove(2);
		}
		else if (inst.size() == 3) {
			inst.remove(0);
		}
		ReconstructionPlan plan = new ObjectMapper().readValue(node.toString(), ReconstructionPlan.class);
		return (ILabeledDataset<?>)plan.reconstructObject();
	}

	public static Map<String, Object> getClassifierFeatureRepresentation(final Classifier c) throws JsonParseException, JsonMappingException, ReconstructionException, IOException {
		Map<String, Object> features = new HashMap<>();
		switch (c.getClass().getSimpleName()) {
		case "PART":
			PART p = (PART)c;
			features.put("a_B", p.getBinarySplits());
			features.put("a_R", p.getReducedErrorPruning());
			features.put("a_N", p.getNumFolds());
			features.put("a_M", p.getMinNumObj());
			break;
		case "J48":
			J48 j48 = (J48)c;
			features.put("a_C", j48.getConfidenceFactor());
			features.put("a_O", j48.getCollapseTree());
			features.put("a_U", j48.getUnpruned());
			features.put("a_B", j48.getBinarySplits());
			features.put("a_J", j48.getUseMDLcorrection());
			features.put("a_S", j48.getSubtreeRaising());
			features.put("a_A", j48.getUseLaplace());
			features.put("a_M", j48.getMinNumObj());
			break;
		case "Logistic":
			Logistic logistic = (Logistic)c;
			features.put("a_R", logistic.getRidge());
			break;
		default:
			throw new UnsupportedOperationException("No feature extraction available for " + c.getClass().getName());
		}
		return features;
	}

	public static Map<String, Object> getClassifierFeatureRepresentation(final String classifierDescription) throws JsonParseException, JsonMappingException, ReconstructionException, IOException {
		Classifier c = getWekaClassifier(classifierDescription).getClassifier();
		return getClassifierFeatureRepresentation(c);
	}

	public static JsonNode getDatasetDescriptionAsJson(final String description) throws IOException {
		String json = description.startsWith("{\"instructions\"") ? description : ( "{\"instructions\":" + description + "}");
		JsonNode node = new ObjectMapper().readTree(json);
		ArrayNode inst = (ArrayNode)node.get("instructions");
		if (inst.size() == 5) {
			inst.remove(0);
			inst.remove(0);
			inst.remove(2);
		}
		else {
			inst.remove(0);
		}
		return inst;
	}

	private static Map<Integer, Map<Double, Map<String, Object>>> featureMapPerDataset = new HashMap<>();

	public static Map<String, Object> getDatasetFeatureRepresentation(final String datasetDescription) throws IOException, ReconstructionException, SplitFailedException, InterruptedException {
		JsonNode node = getDatasetDescriptionAsJson(datasetDescription);
		int id = node.get(0).get("arguments").get(0).asInt();
		//		if (!featureMapPerDataset.containsKey(id)) {
		//			ILabeledDataset<?> data = getData(datasetDescription);
		//			featureMapPerDataset.put(id, getDatasetFeatureRepresentation(data, .7)); // multiply with .7, because the eventual train data is only .7 of the portion here.
		//		}
		//		return featureMapPerDataset.get(id);
		return new DatasetFeatureGenerator("d_").getFeatureRepresentation(SplitterUtil.getLabelStratifiedTrainTestSplit(getData(datasetDescription), 0, .7).get(0));
	}
}
