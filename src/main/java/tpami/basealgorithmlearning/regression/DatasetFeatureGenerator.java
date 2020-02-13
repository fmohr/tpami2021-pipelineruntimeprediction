package tpami.basealgorithmlearning.regression;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.ICategoricalAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

class DatasetFeatureGenerator implements IDatasetFeatureMapper {

	private final String prefix;

	public DatasetFeatureGenerator(final String prefix) {
		super();
		this.prefix = prefix;
	}

	@Override
	public Map<String, Object> getFeatureRepresentation(final ILabeledDataset<?> dataset) {
		Map<String, Object> features = new HashMap<>();
		features.put(this.prefix + "_instances", dataset.size());
		features.put(this.prefix + "_numattributes", dataset.getNumAttributes());
		features.put(this.prefix + "_numlabels", Arrays.stream(dataset.getLabelVector()).collect(Collectors.toSet()).size());
		features.put(this.prefix + "_numnumericattributes", dataset.getInstanceSchema().getAttributeList().stream().filter(a -> a instanceof INumericAttribute).count());
		features.put(this.prefix + "_numsymbolicattributes", dataset.getInstanceSchema().getAttributeList().stream().filter(a -> !(a instanceof INumericAttribute)).count());
		int valuesOfCategoricFeatures = 0;
		for (IAttribute att : dataset.getListOfAttributes()) {
			if (att instanceof ICategoricalAttribute) {
				valuesOfCategoricFeatures += ((ICategoricalAttribute)att).getNumberOfCategories();
			}
		}
		features.put(this.prefix + "_numberofcategories", valuesOfCategoricFeatures);
		//		Map<String, Object> expansions = new HashMap<>();
		//		for (Entry<String, Object> featureEntry : features.entrySet()) {
		//			expansions.put(featureEntry.getKey() + "_2", Math.pow(Double.valueOf(featureEntry.getValue().toString()), 2));
		//			expansions.put(featureEntry.getKey() + "_l", Math.log(Double.valueOf(featureEntry.getValue().toString())));
		//		}
		//		features.putAll(expansions);
		//
		//		/* compute all sub-sets of features */
		//		try {
		//			Collection<Collection<String>> featureSubSets = SetUtil.powerset(features.keySet());
		//			for (Collection<String> subset : featureSubSets) {
		//				if (subset.size() > 3) {
		//					continue;
		//				}
		//				StringBuilder featureName = new StringBuilder("x");
		//				double featureValue = 1;
		//				for (String feature : subset.stream().sorted().collect(Collectors.toList())) {
		//					featureName.append("_" + feature);
		//					featureValue *= Double.valueOf(features.get(feature).toString());
		//				}
		//				features.put(featureName.toString(), featureValue);
		//			}
		//		} catch (InterruptedException e) {
		//			e.printStackTrace();
		//		}
		return features;
	}
}
