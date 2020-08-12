package tpami.safeguard.util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.api4.java.datastructure.kvstore.IKVStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreUtil;
import ai.libs.jaicore.components.model.Component;
import ai.libs.jaicore.components.model.ComponentInstance;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class DataBasedComponentPredictorUtil {

	private static final String[] COMPONENT_IDS = { "multilayerperceptron", "bayesnet", "decisionstump", "decisiontable", "ibk", "logistic", "naivebayes", "oner", "part", "randomforest", "randomtree", "smo", "votedperceptron", "kstar",
			"reptree", "simplelogistic", "j48", "lmt", "jrip", "naivebayesmultinomial", "zeror", "adaboostm1", "bagging", "logitboost", "randomcommittee", "randomsubspace" };
	public static final Class<?>[] WEKA_CLASSES = { MultilayerPerceptron.class, BayesNet.class, DecisionStump.class, DecisionTable.class, IBk.class, Logistic.class, NaiveBayes.class, OneR.class, PART.class, RandomForest.class,
			RandomTree.class, SMO.class, VotedPerceptron.class, KStar.class, REPTree.class, SimpleLogistic.class, J48.class, LMT.class, JRip.class, NaiveBayesMultinomial.class, ZeroR.class, AdaBoostM1.class, Bagging.class, LogitBoost.class,
			RandomCommittee.class, RandomSubSpace.class };

	private static final Logger LOGGER = LoggerFactory.getLogger(DataBasedComponentPredictorUtil.class);
	private static final Map<String, String> ID2WekaMap = new HashMap<>();
	private static final Map<String, String> Weka2IDMap = new HashMap<>();
	private static Set<String> alreadyWarned = new HashSet<>();

	public static Component getComponentForID(final String id) {
		return null;
	}

	public static final List<String> PREPROCESSORS = Arrays.asList("bestfirst_cfssubseteval", "greedystepwise_cfssubseteval", "ranker_correlationattributeeval", "ranker_gainratioattributeeval", "ranker_infogainattributeeval",
			"ranker_onerattributeeval", "ranker_principalcomponents", "ranker_relieffattributeeval", "ranker_symmetricaluncertattributeeval");

	public static String componentInstanceToPreprocessorID(final ComponentInstance ci) {
		ComponentInstance searcher = ci.getSatisfactionOfRequiredInterfaces().get("search");
		ComponentInstance evaluator = ci.getSatisfactionOfRequiredInterfaces().get("eval");

		switch (evaluator.getComponent().getName()) {
		case "weka.attributeSelection.CfsSubsetEval":
			switch (searcher.getComponent().getName()) {
			case "weka.attributeSelection.BestFirst":
				return "bestfirst_cfssubseteval";
			case "weka.attributeSelection.GreedyStepwise":
				return "greedystepwise_cfssubseteval";
			default:
				return null;
			}
		case "weka.attributeSelection.CorrelationAttributeEval":
			return "ranker_correlationattributeeval";
		case "weka.attributeSelection.GainRatioAttributeEval":
			return "ranker_gainratioattributeeval";
		case "weka.attributeSelection.InfoGainAttributeEval":
			return "ranker_infogainattributeeval";
		case "weka.attributeSelection.OneRAttributeEval":
			return "ranker_onerattributeeval";
		case "weka.attributeSelection.PrincipalComponents":
			return "ranker_principalcomponents";
		case "weka.attributeSelection.ReliefFAttributeEval":
			return "ranker_relieffattributeeval";
		case "weka.attributeSelection.SymmetricalUncertAttributeEval":
			return "ranker_symmetricaluncertattributeeval";
		default:
			return null;
		}
	}

	private static void loadID2WekaMapping() {
		for (int i = 0; i < COMPONENT_IDS.length; i++) {
			ID2WekaMap.put(COMPONENT_IDS[i], WEKA_CLASSES[i].getName());
			Weka2IDMap.put(WEKA_CLASSES[i].getName(), COMPONENT_IDS[i]);
		}
	}

	public static String mapID2Weka(final String id) {
		if (ID2WekaMap.isEmpty()) {
			loadID2WekaMapping();
		}

		if (ID2WekaMap.containsKey(id)) {
			return ID2WekaMap.get(id);
		}

		if (!alreadyWarned.contains(id)) {
			LOGGER.debug("Could not find mapping for id {}", id);
			alreadyWarned.add(id);
		}
		return id;
	}

	public static String mapWeka2ID(final String wekaClassName) {
		if (Weka2IDMap.containsKey(wekaClassName)) {
			return Weka2IDMap.get(wekaClassName);
		}
		return wekaClassName;
	}

	public static KVStoreCollection readCSV(final File csvFile, final Map<String, String> commonFields) throws IOException {
		alreadyWarned.clear();
		if (!csvFile.exists()) {
			throw new IllegalArgumentException("CSV file " + csvFile + " does not exist.");
		}

		KVStoreCollection col = KVStoreUtil.readFromCSVWithHeader(csvFile, commonFields, ",");
		col.setCollectionID(csvFile.getName());
		return col;
	}

	public static Instances kvStoreCollectionToWekaInstances(final KVStoreCollection col, final String targetAttributeName, final Collection<String> featureAttributeNames) {
		return kvStoreCollectionToWekaInstances(col, targetAttributeName, featureAttributeNames.toArray(new String[] {}));
	}

	public static Instances kvStoreCollectionToWekaInstances(final KVStoreCollection col, final String targetAttributeName, final String... featureAttributeNames) {
		Map<String, String> attributeTypes = new HashMap<>();
		Map<String, Set<String>> valuesOfAttributes = new HashMap<>();
		for (IKVStore store : col) {
			for (String featureAttributeName : featureAttributeNames) {
				valuesOfAttributes.computeIfAbsent(featureAttributeName, t -> new HashSet<>()).add(store.getAsString(featureAttributeName));
			}
			valuesOfAttributes.computeIfAbsent(targetAttributeName, t -> new HashSet<>()).add(store.getAsString(targetAttributeName));
		}

		for (Entry<String, Set<String>> attValues : valuesOfAttributes.entrySet()) {
			boolean allInt = true;
			boolean allDouble = true;
			for (String v : attValues.getValue()) {
				if (v.trim().isEmpty()) {
					continue;
				}
				try {
					Integer.parseInt(v);
				} catch (Exception e) {
					allInt = false;
				}
				try {
					Double.parseDouble(v);
				} catch (Exception e) {
					allDouble = false;
				}

				if (!(allInt || allDouble)) {
					break;
				}
			}
			if (allInt || allDouble) {
				attributeTypes.put(attValues.getKey(), "numeric");
			} else {
				attributeTypes.put(attValues.getKey(), "nominal");
			}
		}

		Map<String, Attribute> attributes = new HashMap<>();
		for (Entry<String, String> entry : attributeTypes.entrySet()) {
			switch (entry.getValue()) {
			case "numeric":
				attributes.put(entry.getKey(), new Attribute(entry.getKey()));
				break;
			case "nominal":
				attributes.put(entry.getKey(), new Attribute(entry.getKey(), new ArrayList<>(valuesOfAttributes.get(entry.getKey()))));
				break;
			}
		}

		return kvStoreCollectionToWekaInstances(col, attributes, targetAttributeName, featureAttributeNames);
	}

	public static Instances kvStoreCollectionToWekaInstances(final KVStoreCollection col, final Map<String, Attribute> attributeTypes, final String targetAttributeName, final String... featureAttributeNames) {
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (String feature : featureAttributeNames) {
			attributes.add(attributeTypes.get(feature));
		}
		attributes.add(attributeTypes.get(targetAttributeName));

		Instances data = new Instances(col.getCollectionID() + "-" + targetAttributeName, attributes, col.size());
		for (IKVStore store : col) {
			try {
				Instance newInst = new DenseInstance(data.numAttributes());
				for (Attribute att : attributes) {
					if (att.isNumeric()) {
						try {
							newInst.setValue(att, store.getAsDouble(att.name()));
						} catch (NumberFormatException e) {
							if (att.name().equals("applicationtime")) {
								continue;
							}
							throw e;
						}
					} else if (att.isNominal()) {
						newInst.setValue(att, store.getAsString(att.name()));
					}
				}
				newInst.setDataset(data);
				data.add(newInst);
			} catch (NumberFormatException e) {

			}
		}
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static String safeGuardComponentToString(final String componentName, final Map<String, Object> containedModels) {
		StringBuilder sb = new StringBuilder();
		sb.append(componentName);
		sb.append(" [");
		sb.append(containedModels.entrySet().stream().map(x -> x.getKey() + ":" + (x.getValue() != null)).collect(Collectors.joining(",")));
		sb.append("]");
		return sb.toString();
	}

	public static boolean isPreprocessor(final String asString) {
		return false;
	}

}
