package tpami.safeguard.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.ICategoricalAttribute;
import org.api4.java.ai.ml.core.dataset.schema.attribute.INumericAttribute;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tpami.safeguard.api.EMetaFeature;
import tpami.safeguard.api.IMetaFeatureContainer;

public class MetaFeatureContainer implements IMetaFeatureContainer {

	private static final Logger LOGGER = LoggerFactory.getLogger(MetaFeatureContainer.class);

	private final Map<EMetaFeature, Double> featureMap;
	private ILabeledDataset<?> dataset = null;

	private static double numAttributesAfterBinarization(final ILabeledDataset<?> dataset) {
		double numAttributes = 0.0;
		for (IAttribute att : dataset.getListOfAttributes()) {
			if (att instanceof INumericAttribute) {
				numAttributes += 1.0;
			} else if (att instanceof ICategoricalAttribute) {
				numAttributes += ((ICategoricalAttribute) att).getNumberOfCategories();
			} else {
				LOGGER.error("Unsupported attribute type in the dataset");
			}
		}
		return numAttributes;
	}

	public MetaFeatureContainer(final ILabeledDataset<?> dataset) {
		this(dataset.size(), numAttributesAfterBinarization(dataset));
		this.dataset = dataset;
	}

	public MetaFeatureContainer(final double numInstances, final double numAttributes) {
		this.featureMap = new HashMap<>();
		this.featureMap.put(EMetaFeature.NUM_INSTANCES, numInstances);
		this.featureMap.put(EMetaFeature.NUM_ATTRIBUTES, numAttributes);
	}

	public MetaFeatureContainer(final double numInstances, final double numAttributes, final Map<EMetaFeature, Double> additionalFeatures) {
		this(numInstances, numAttributes);
		this.featureMap.putAll(additionalFeatures);
	}

	@Override
	public ILabeledDataset<?> getDataset() {
		return this.dataset;
	}

	@Override
	public double getFeature(final EMetaFeature feature) {
		return this.featureMap.get(feature);
	}

	/**
	 * Turn the meta features of this container into a double array representation.
	 * @return A double array representing the meta features within this container.
	 */
	@Override
	public double[] toFeatureVector() {
		List<EMetaFeature> keys = new ArrayList<>(this.featureMap.keySet());
		Collections.sort(keys);
		return keys.stream().mapToDouble(x -> this.featureMap.get(x)).toArray();
	}

	@Override
	public boolean equals(final Object other) {
		if (other instanceof MetaFeatureContainer) {
			return new EqualsBuilder().append(this.dataset, ((MetaFeatureContainer) other).dataset).append(this.featureMap, ((MetaFeatureContainer) other).featureMap).isEquals();
		}
		return false;
	}

	@Override
	public int hashCode() {
		return new HashCodeBuilder().append(this.dataset).append(this.featureMap).toHashCode();
	}

	@Override
	public String toString() {
		return this.featureMap.toString();
	}

}
