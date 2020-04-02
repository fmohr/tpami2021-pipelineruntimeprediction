package tpami.basealgorithmlearning.regression;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.ICategoricalAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

public class BasicDatasetFeatureGenerator implements IDatasetFeatureMapper {

	private String prefix = "";
	private String suffix = "";

	public BasicDatasetFeatureGenerator() {

	}

	public BasicDatasetFeatureGenerator(final String prefix) {
		this();
		this.setPrefix(prefix);
	}

	@Override
	public Map<String, Object> getFeatureRepresentation(final ILabeledDataset<?> dataset) {
		Map<String, Object> features = new HashMap<>();
		features.put(this.prefix + "numinstances" + this.suffix, dataset.size());
		features.put(this.prefix + "numattributes"+ this.suffix, dataset.getNumAttributes());
		features.put(this.prefix + "numlabels"+ this.suffix, Arrays.stream(dataset.getLabelVector()).collect(Collectors.toSet()).size());
		features.put(this.prefix + "numnumericattributes"+ this.suffix, dataset.getInstanceSchema().getAttributeList().stream().filter(a -> a instanceof INumericAttribute).count());
		features.put(this.prefix + "numsymbolicattributes"+ this.suffix, dataset.getInstanceSchema().getAttributeList().stream().filter(a -> !(a instanceof INumericAttribute)).count());

		int valuesOfCategoricFeatures = 0;
		int attributesAddedByBinarization = 0;
		for (IAttribute att : dataset.getListOfAttributes()) {
			if (att instanceof ICategoricalAttribute) {
				int numCategories = ((ICategoricalAttribute)att).getNumberOfCategories();
				valuesOfCategoricFeatures += numCategories;
				if (numCategories > 2) {
					attributesAddedByBinarization += numCategories; // one is subtracted, because the original attribute is removed!
				}
				else {
					attributesAddedByBinarization ++;
				}
			}
		}
		features.put(this.prefix + "numberofcategories"+ this.suffix, valuesOfCategoricFeatures);
		features.put(this.prefix + "numericattributesafterbinarization"+ this.suffix, (long)features.get(this.prefix + "numnumericattributes"+ this.suffix) + attributesAddedByBinarization);

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

	public String getPrefix() {
		return this.prefix;
	}

	public void setPrefix(final String prefix) {
		this.prefix = prefix;
	}

	public String getSuffix() {
		return this.suffix;
	}

	public void setSuffix(final String suffix) {
		this.suffix = suffix;
	}
}
