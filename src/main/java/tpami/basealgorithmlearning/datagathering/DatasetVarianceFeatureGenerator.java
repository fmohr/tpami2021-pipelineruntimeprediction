package tpami.basealgorithmlearning.datagathering;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.nd4j.linalg.api.ndarray.INDArray;

import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;

public class DatasetVarianceFeatureGenerator implements IDatasetFeatureMapper {

	private String prefix = "";
	private String suffix = "";

	public DatasetVarianceFeatureGenerator() {

	}

	public DatasetVarianceFeatureGenerator(final String prefix) {
		this();
		this.setPrefix(prefix);
	}

	private INDArray datasetToArray(final ILabeledDataset<?> dataset) throws Exception {
		return WekaUtil.instances2matrix(new WekaInstances(dataset).getInstances());
	}

	@Override
	public Map<String, Object> getFeatureRepresentation(final ILabeledDataset<?> dataset) throws Exception {
		Map<String, Object> features = new HashMap<>();
		INDArray matrix = this.datasetToArray(dataset);
		INDArray var = matrix.var(0);

		/* compute total variance */
		double totalVariance = 0;
		List<Double> variances = new ArrayList<>();
		for (double v : var.toDoubleVector()) {
			totalVariance += v;
			variances.add(v);
		}
		features.put(this.prefix + "totalvariance" + this.suffix, totalVariance);

		/* compute number of attribute required to capture 50%, 90%, and 95% of the variance */
		double vCoverage = 0;
		int cov50 = -1;
		int cov90 = -1;
		int cov95 = -1;
		int cov99 = -1;
		List<Double> varianceInDescendingOrder = variances.stream().sorted().collect(Collectors.toList());
		Collections.reverse(varianceInDescendingOrder);
		int consideredAtts = 0;
		for (double v : varianceInDescendingOrder) {
			vCoverage += v;
			consideredAtts++;
			if (cov50 < 0 && vCoverage / totalVariance >= .5) {
				cov50 = consideredAtts;
			}
			if (cov90 < 0 && vCoverage / totalVariance >= .9) {
				cov90 = consideredAtts;
			}
			if (cov95 < 0 && vCoverage / totalVariance >= .95) {
				cov95 = consideredAtts;
			}
			if (cov99 < 0 && vCoverage / totalVariance >= .99) {
				cov99 = consideredAtts;
			}
		}
		features.put(this.prefix + "attributestocover50pctvariance" + this.suffix, cov50);
		features.put(this.prefix + "attributestocover90pctvariance" + this.suffix, cov90);
		features.put(this.prefix + "attributestocover95pctvariance" + this.suffix, cov95);
		features.put(this.prefix + "attributestocover99pctvariance" + this.suffix, cov99);

		/* compute absolute relative accumulated variances */
		int n = 10;
		double accVar = 0;
		for (int i = 0; i < n; i++) {
			double nextBiggestVariance = varianceInDescendingOrder.isEmpty() ? 0 : varianceInDescendingOrder.get(0);
			if (!varianceInDescendingOrder.isEmpty()) {
				varianceInDescendingOrder.remove(0);
			}
			accVar += nextBiggestVariance;
			features.put(this.prefix + "accvarianceabs" + (i + 1) + this.suffix, accVar);
			features.put(this.prefix + "accvariancerel" + (i + 1) + this.suffix, accVar / totalVariance);
		}

		/* compute relative accumulated variances */
		// int nCov = 10;
		// double accCovar = 0;
		// for (int i = 0; i < nCov; i++) {
		// int entry = cov.argMax().getInt(0);
		// double nextBiggestCovariance = cov.getDouble(entry);
		// cov.putScalar(entry, 0);
		// accCovar += nextBiggestCovariance;
		// features.put(this.prefix + "acccovarianceabs" + (i+1) + this.suffix, accCovar);
		// features.put(this.prefix + "acccovariancerel" + (i+1) + this.suffix, accCovar / absoluteLinearRelationShip);
		// }
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

	public static void main(final String[] args) throws Exception {
		ILabeledDataset<?> ds = OpenMLDatasetReader.deserializeDataset(1457);
		DatasetVarianceFeatureGenerator gen = new DatasetVarianceFeatureGenerator();
		System.out.println(ds.size() + " x " + ds.getNumAttributes());
		Map<String, Object> features = gen.getFeatureRepresentation(ds);
		for (String key : features.keySet().stream().sorted().collect(Collectors.toList())) {
			System.out.println(key + ": " + features.get(key));
		}
	}
}
