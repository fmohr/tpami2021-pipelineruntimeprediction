package tpami.basealgorithmlearning.regression;

import java.util.Map;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;

public interface IDatasetFeatureMapper {
	public Map<String, Object> getFeatureRepresentation(ILabeledDataset<?> dataset) throws Exception;
}
