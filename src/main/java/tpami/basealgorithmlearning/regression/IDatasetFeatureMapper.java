package tpami.basealgorithmlearning.regression;

import java.util.Map;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

interface IDatasetFeatureMapper {
	public Map<String, Object> getFeatureRepresentation(ILabeledDataset<?> dataset);
}
