package tpami.basealgorithmlearning.datagathering;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Random;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.core.dataset.Dataset;
import ai.libs.jaicore.ml.core.dataset.DenseInstance;
import ai.libs.jaicore.ml.core.dataset.schema.LabeledInstanceSchema;
import ai.libs.jaicore.ml.core.dataset.schema.attribute.NumericAttribute;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;

public class ExperimentUtil implements ILoggingCustomizable {

	private static final File folder = new File("data/SMOTE");

	private Logger logger = LoggerFactory.getLogger(ExperimentUtil.class);

	public static boolean doesExtensionExist(final int openmlid) {
		return getFileForSMOTEExtension(openmlid).exists();
	}

	public static int getMaximumInstancesManagableForDataset(final ILabeledDataset<?> ds) {
		return (int) Math.floor((5 * Math.pow(10, 8)) / ds.getNumAttributes());
	}

	public static File getFileForSMOTEExtension(final int openmlid) {
		return new File(folder + File.separator + openmlid + ".arff");
	}

	public void createSMOTEExtensionOfOpenMLDataset(final int id, final int size) throws Exception {
		ILabeledDataset<?> origDS = OpenMLDatasetReader.deserializeDataset(id);
		ILabeledDataset<?> ds = this.createSMOTEExtensionOfDataset(origDS, size);

		int duplicates = 0;
		for (ILabeledInstance i : ds) {
			if (origDS.contains(i)) {
				duplicates++;
			}
		}
		if (duplicates > 0) {
			throw new IllegalStateException("Create a duplicate, which must not happen!");
		}
		ArffDatasetAdapter.serializeDataset(getFileForSMOTEExtension(id), ds);
	}

	public ILabeledDataset<?> createSMOTEExtensionOfDataset(final ILabeledDataset<?> originalDataset, final int sizeOfExtension) throws Exception {

		/* determine how many instance per class we want to see in the end */
		Instances inst = new WekaInstances(originalDataset).getInstances();
		Map<String, Integer> absoluteClassFrequencies = WekaUtil.getNumberOfInstancesPerClass(inst);
		Map<String, Integer> expectedClassFrequencies = new HashMap<>();
		int numInstances = inst.size();
		int sumExpected = 0;
		for (Entry<String, Integer> entry : absoluteClassFrequencies.entrySet()) {
			int expected = (int) Math.floor(entry.getValue() * sizeOfExtension * 1.0 / numInstances);
			sumExpected += expected;
			expectedClassFrequencies.put(entry.getKey(), expected);
		}
		for (Entry<String, Integer> entry : absoluteClassFrequencies.entrySet()) {
			if (sumExpected == sizeOfExtension) {
				break;
			}
			expectedClassFrequencies.put(entry.getKey(), expectedClassFrequencies.get(entry.getKey()) + 1);
			sumExpected++;
		}

		/* start SMOTE */
		Instances upsampled = inst;
		this.logger.info("Now producing a dataset of {} instances with SMOTE.", sizeOfExtension);
		List<String> classValues = new ArrayList<>(WekaUtil.getClassesActuallyContainedInDataset(inst));
		while (upsampled.size() - inst.size() < sizeOfExtension) {
			int numberOfMissingInstances = 10 + sizeOfExtension - upsampled.size() + inst.size(); // 10 is just a buffer to avoid stupid cases of 2 more required instances
			this.logger.info("Next SMOTE iteration round. Available instances: {}. Missing instances: {}", upsampled.size(), numberOfMissingInstances);

			/* in each round, consider each class as the minority class once */
			double percentage = (numberOfMissingInstances * 100.0 / upsampled.size());
			percentage *= 2; // this is because the algorithm does not really respect the required number.
			double percentageForEachClass = percentage;
			this.logger.info("Setting percentage to {}, which is {} for each class.", percentage, percentageForEachClass);

			for (String classValue : classValues) { // the class value is not used because there is a bug in SMOTE that does not allow to do so!
				SMOTE smote = new SMOTE();
				smote.setPercentage(percentageForEachClass);
				smote.setInputFormat(upsampled);
				upsampled = Filter.useFilter(upsampled, smote);
			}
		}
		this.logger.info("Gathered enough instances. Now creating a stratified sub-sample with the following distribution: {}", expectedClassFrequencies);

		/* first erase the first entries belonging to the original data */
		int k = 0;
		for (Instance i : inst) {
			String serializedInstance = i.toString();
			int n = upsampled.size();
			for (int j = 0; j < n; j++) {
				Instance iSMOTE = upsampled.get(j);
				if (serializedInstance.equals(iSMOTE.toString())) {
					upsampled.remove(j);
					j--;
					n--;
				}
			}
			k++;
			this.logger.info("{}. Remaining instances: {}/{}", k * 100.0 / inst.size(), upsampled.size(), sizeOfExtension);
			if (upsampled.size() == sizeOfExtension) {
				this.logger.info("Stopping duplicate elimination since we now would get to duplicates anyway.");
				break;
			}
		}

		/* now get the required number of instances for each class */
		Instances output = new Instances(inst);
		output.clear();
		for (Entry<String, Instances> entry : WekaUtil.getInstancesPerClass(upsampled).entrySet()) {
			int n = expectedClassFrequencies.get(entry.getKey());
			int underCoverage = (int) Math.floor(n * 1.0 / entry.getValue().size());
			this.logger.debug("Undercoverage is {} for {} instances.", underCoverage, entry.getValue().size());
			for (int i = 0; i < underCoverage; i++) {
				output.addAll(entry.getValue());
			}
			int remainingMissingInstances = n - underCoverage * entry.getValue().size();
			output.addAll(entry.getValue().stream().limit(remainingMissingInstances).collect(Collectors.toList()));
		}
		this.logger.info("Finished SMOTE sampling. Distribution of classes in extension is: {}", WekaUtil.getNumberOfInstancesPerClass(output));
		return new WekaInstances(output);
	}

	public List<ILabeledDataset<?>> createSizeAdjustedTrainTestSplit(final int openmlid, final int trainsize, final int minTestSize, final int numAttributes, final Random random) throws Exception {
		ILabeledDataset<?> dsOrig = OpenMLDatasetReader.deserializeDataset(openmlid);
		return this.createSizeAdjustedTrainTestSplit(dsOrig, openmlid, trainsize, minTestSize, numAttributes, random);
	}

	public ILabeledDataset<?> getSMOTEDatasetExtension(final int openmlid) throws DatasetDeserializationFailedException {
		return ArffDatasetAdapter.readDataset(getFileForSMOTEExtension(openmlid));
	}

	public List<ILabeledDataset<?>> createSizeAdjustedTrainTestSplit(final ILabeledDataset<?> originalDataset, final int openmlid, final int trainsize, final int minTestSize, final int numAttributes, final Random random) throws Exception {
		if (originalDataset.size() < trainsize + minTestSize && !doesExtensionExist(openmlid)) {
			throw new IllegalArgumentException("Dataset has not sufficient points (" +  originalDataset.size() + " instead of the required " + (trainsize + minTestSize) + "), but no SMOTE file existent for openmlid " + openmlid);
		}
		return this.createSizeAdjustedTrainTestSplit(originalDataset, originalDataset.size() < trainsize ? this.getSMOTEDatasetExtension(openmlid) : null, trainsize, minTestSize, numAttributes, random);
	}

	public List<ILabeledDataset<?>> createSizeAdjustedTrainTestSplit(final ILabeledDataset<?> originalDataset, final ILabeledDataset<?> datasetExtension, final int trainsize, final int minTestSize, final int numAttributes,
			final Random random) throws Exception {

		/* first make sure that the dataset size is at least the required train size + the min test size. If necessary, apply SMOTE to get more instances */
		int minimumRequiredInstances = trainsize + minTestSize;
		this.logger.info("Creating a split for {} = {} + {} data points. Number of existing points in data: {}", minimumRequiredInstances, trainsize, minTestSize, originalDataset.size());

		/* if the dataset has enough attributes, just randomly remove some of them */
		if (originalDataset.getNumAttributes() > numAttributes) {
			this.logger.info("Removing {}/{} attributes.", originalDataset.getNumAttributes() - numAttributes, originalDataset.getNumAttributes());
			Collection<IAttribute> attsToRemove = SetUtil.getRandomSetOfIntegers(originalDataset.getNumAttributes(), originalDataset.getNumAttributes() - numAttributes, random).stream().map(i -> originalDataset.getAttribute(i))
					.collect(Collectors.toList());
			attsToRemove.forEach(a -> {
				originalDataset.removeColumn(a);
				if (datasetExtension != null) {
					datasetExtension.removeColumn(a);
				}
			});
			this.logger.info("New dataset format is {}x{} for original data and {}x{} for SMOTE data", originalDataset.size(), originalDataset.getNumAttributes(), datasetExtension != null ? datasetExtension.size() : null, datasetExtension != null ? datasetExtension.getNumAttributes() : null);
		}

		/* if the dataset is large enough to provide all necessary instances for the training set, use the complementary data only for testing */
		List<ILabeledDataset<?>> split;
		if (originalDataset.size() >= trainsize) {
			split = SplitterUtil.getLabelStratifiedTrainTestSplit(originalDataset, random, 1.0 * trainsize / originalDataset.size());

			/* if the dataset is not large enough to provide all instances for testing, fille the test fold with complementary instances. */
			if (originalDataset.size() < minimumRequiredInstances) {
				if (datasetExtension != null) {
					ILabeledDataset<ILabeledInstance> testFold = (ILabeledDataset<ILabeledInstance>) split.get(1);
					testFold.addAll(SetUtil.getRandomSubset(datasetExtension, minTestSize - testFold.size(), random));
				}
				else {
					throw new IllegalStateException("Original data has not enough instances, and there is no SMOTE file available!");
				}
			}
		}
		else {

			/* if the dataset is not even large enough to create the training fold with meaningful randomization, first create one huge joint dataset and then sample from this one */
			Objects.requireNonNull(datasetExtension, "Original data has not enough instances, and there is no SMOTE file available for dataset!");
			ILabeledDataset dsJoin = (ILabeledDataset) originalDataset.createCopy();
			dsJoin.addAll(datasetExtension);
			if (dsJoin.size() < trainsize) {
				throw new IllegalStateException("Joint dataset should have at least " + trainsize + " many datapoints, but has only " + dsJoin.size());
			}
			split = SplitterUtil.getLabelStratifiedTrainTestSplit(dsJoin, random, 1.0 * trainsize / dsJoin.size());
		}

		if (originalDataset.getNumAttributes() < numAttributes) {
			int requiredColumns = numAttributes - originalDataset.getNumAttributes();
			this.logger.info("Creating {} artificial columns.", requiredColumns);
			List<IAttribute> attributes = originalDataset.getListOfAttributes();
			for (int j = 0; j < requiredColumns; j++) {
				attributes.add(new NumericAttribute("artf_" + j));
			}
			LabeledInstanceSchema extendedScheme = new LabeledInstanceSchema(originalDataset.getRelationName(), attributes, originalDataset.getLabelAttribute());
			Dataset fold1New = new Dataset(extendedScheme);
			int m = split.get(0).size();
			int count = 0;
			for (ILabeledInstance i : split.get(0)) {
				count ++;
				List<Object> values = new ArrayList<>(Arrays.asList(i.getAttributes()));
				for (int j = 0; j < requiredColumns; j++) {
					values.add(random.nextDouble());
				}
				DenseInstance newInst = new DenseInstance(values, i.getLabel());
				fold1New.add(newInst);
				this.logger.debug("{}/{}", count, m);
			}
			Dataset fold2New = new Dataset(extendedScheme);
			m = minTestSize;
			count = 0;
			for (ILabeledInstance i : split.get(1)) {
				count ++;
				List<Object> values = new ArrayList<>(Arrays.asList(i.getAttributes()));
				for (int j = 0; j < requiredColumns; j++) {
					values.add(random.nextDouble());
				}
				DenseInstance newInst = new DenseInstance(values, i.getLabel());
				fold2New.add(newInst);
				this.logger.debug("{}/{}", count, m);
				if (count == minTestSize) {
					break;
				}
			}
			split = Arrays.asList(fold1New, fold2New);
		}
		this.logger.info("Produced two folds. First has dimension {}x{}, second has dimension {}x{}", split.get(0).size(), split.get(0).getNumAttributes(), split.get(1).size(), split.get(1).getNumAttributes());
		return split;
	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger = LoggerFactory.getLogger(name);
	}

	public static void main(final String[] args) throws Exception {
		ExperimentUtil u = new ExperimentUtil();
		u.setLoggerName(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);
		List<ILabeledDataset<?>> split = u.createSizeAdjustedTrainTestSplit(6, 10000, 1500, 10000, new Random(0));
		ILabeledDataset<?> trainFold = split.get(0);
		System.out.println(trainFold.getNumAttributes() * trainFold.size());
	}
}
