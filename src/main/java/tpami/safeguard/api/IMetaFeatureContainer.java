package tpami.safeguard.api;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

/**
 * Container for a meta feature description of a dataset.
 *
 * @author mwever
 */
public interface IMetaFeatureContainer {

	public ILabeledDataset<?> getDataset();

	public double getFeature(final EMetaFeature feature);

	public double[] toFeatureVector();

}
