package tpami.basealgorithmlearning.datagathering;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.common.control.ILoggingCustomizable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.MathExt;
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

	public static final File folder = new File("data/SMOTE");
	public static final List<Integer> LEVELS = Arrays.asList(3000, 7000, 12000, 22000, 52000, 102000, 502000, 1002000); // this always includes 2000 test instances

	private Logger logger = LoggerFactory.getLogger(ExperimentUtil.class);

	public boolean doesExtensionExist(final int openmlid, final int level) {
		return this.getFileForSMOTEExtension(openmlid, level).exists();
	}

	public static int getMaximumAttributesManagableForNumberOfInstances(final int numInstances) {
		return numInstances <= 3000 ? (int) Math.pow(10, 5) : (int) Math.ceil((5 * Math.pow(10, 8)) / (numInstances - 1500));
	}

	// public static int getMaximumInstancesManagableForDataset(final ILabeledDataset<?> ds) {
	// return (int) Math.floor((5 * Math.pow(10, 8)) / ds.getNumAttributes());
	// }

	public int getMaxLevelForWhichASMOTEFileCanBeWrittenWithoutPruning(final int openmlid) {
		try {
			int numAttributes = OpenMLDatasetReader.getNumberOfAttributes(openmlid);
			return LEVELS.stream().filter(i -> getMaximumAttributesManagableForNumberOfInstances(i) >= numAttributes).mapToInt(i -> i).max().getAsInt();
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public int getExtensionRowsForNumInstancens(final int numInstances) {
		return LEVELS.stream().filter(i -> i >= numInstances).mapToInt(i -> i).min().getAsInt();
	}

	public int getExtensionColsForNumInstancens(final int numInstances) {
		return getMaximumAttributesManagableForNumberOfInstances(this.getExtensionRowsForNumInstancens(numInstances));
	}

	public File getFileForSMOTEExtension(final int openmlid, final int numInstances) {
		//		int maxLevelWithoutPruning = this.getMaxLevelForWhichASMOTEFileCanBeWrittenWithoutPruning(openmlid);
		int level;
		//		if (maxLevelWithoutPruning >= numInstances) {
		//			this.logger.debug("Given the number of attributes of dataset {} and a desired number of {} instances, a SMOTE file for size/level {} is sufficient.", openmlid, numInstances, maxLevelWithoutPruning);
		//			level = maxLevelWithoutPruning;
		//		}
		//		else {
		level = this.getExtensionRowsForNumInstancens(numInstances);
		//		}
		this.logger.debug("Identified level {} for number of instances {}", level, numInstances);
		return new File(folder + File.separator + level + File.separator + openmlid + ".arff");
	}

	List<Integer> getIgnoredColumnsForDatasetAndGivenNumberTotalInstances(final int openmlid, final int totalInstances) throws DatasetDeserializationFailedException {
		return this.getIgnoredColumnsForDatasetAndGivenNumberTotalInstances(OpenMLDatasetReader.deserializeClassificationDataset(openmlid, Arrays.asList()), totalInstances);
	}

	List<Integer> getIgnoredColumnsForDatasetAndGivenNumberTotalInstances(final ILabeledDataset<?> dsOrig, final int totalInstances) throws DatasetDeserializationFailedException {
		int possibleAttributes = getMaximumAttributesManagableForNumberOfInstances(totalInstances);
		return this.sortColumnsByRelevance(dsOrig, true).stream().limit(Math.max(0, dsOrig.getNumAttributes() - possibleAttributes)).collect(Collectors.toList());
	}

	List<Integer> getIgnoredColumnsForDatasetAndGivenNumberOfMaxRemainingAttributes(final ILabeledDataset<?> dsOrig, final int numInstances, final int maxAttributes) throws DatasetDeserializationFailedException {
		int numAttributesInOriginalDataset = dsOrig.getNumAttributes();
		int numOriginalAttributesToIgnore = Math.max(0, numAttributesInOriginalDataset - maxAttributes);

		List<Integer> ignoredColumns = new ArrayList<>();

		/* first add the attributes of the original dataset, which should be ignored */
		if (numOriginalAttributesToIgnore > 0) {
			ignoredColumns.addAll(this.sortColumnsByRelevance(dsOrig, true).stream().limit(numOriginalAttributesToIgnore).collect(Collectors.toList()));
		}

		/* now add artificial attributes that should be ignored */
		int numAttributesInRelevantExtension = this.getExtensionColsForNumInstancens(numInstances);
		int numAttributesToIgnoreInRelevantExentsion = numAttributesInRelevantExtension - maxAttributes;
		int numArtificialAttributeToIgnore = Math.max(0, numAttributesToIgnoreInRelevantExentsion - numOriginalAttributesToIgnore);
		for (int i = 0; i < numArtificialAttributeToIgnore; i++) {
			ignoredColumns.add(numAttributesInOriginalDataset + i);
		}
		return ignoredColumns;
	}

	public WekaInstances getDatasetWithReducedNumberOfColumnsForGivenNumberOfTotalInstances(final int openmlid, final int totalInstances) throws DatasetDeserializationFailedException {
		List<Integer> ignoredColumns = this.getIgnoredColumnsForDatasetAndGivenNumberTotalInstances(openmlid, totalInstances);
		return new WekaInstances(OpenMLDatasetReader.deserializeClassificationDataset(openmlid, ignoredColumns));
	}

	public List<Integer> sortColumnsByRelevance(final ILabeledDataset<?> ds, final boolean asc) {

		/* compute map with relevances */
		int d = ds.getNumAttributes();
		int n = ds.size();
		Map<Integer, Double> variances = new HashMap<>();
		for (int i = 0; i < d; i++) {
			Object[] col = ((Dataset) ds).getColumn(i);
			//			int counter = 0;
			DescriptiveStatistics stats = new DescriptiveStatistics();
			for (int j = 0; j < n; j++) {
				if (col[j] != null) {
					//					counter++;
					stats.addValue(Double.valueOf("" + col[j]));
				}
			}
			variances.put(i, stats.getVariance());
		}

		/* get list of indices sorted by relevances */
		List<Integer> indices = variances.entrySet().stream().sorted((e1, e2) -> Double.compare(e1.getValue(), e2.getValue())).map(e -> e.getKey()).collect(Collectors.toList());
		if (!asc) {
			Collections.reverse(indices);
		}
		return indices;
	}

	public void createSMOTEExtensionOfOpenMLDatasets(final int id) throws Exception {

		this.logger.debug("Start reading in original dataset {}.", id);
		ILabeledDataset<?> origCompleteDataset = OpenMLDatasetReader.deserializeClassificationDataset(id, Arrays.asList());
		this.logger.debug("Finished reading in dataset {} of size {} x {}", id, origCompleteDataset.size(), origCompleteDataset.getNumAttributes());

		//		int maxLevelThatCanBeSMOTEDWithoutPruningAttributes = this.getMaxLevelForWhichASMOTEFileCanBeWrittenWithoutPruning(id);

		for (int i = 0; i < LEVELS.size(); i++) {
			int numInstances = LEVELS.get(i);
			int numInstancesPreviousSegment = i > 0 ? LEVELS.get(i-1) : 0;
			int numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment = getMaximumAttributesManagableForNumberOfInstances(numInstancesPreviousSegment);
			int numAttributesProvided = origCompleteDataset.getNumAttributes();

			//			if (numInstances < maxLevelThatCanBeSMOTEDWithoutPruningAttributes) {
			//				this.logger.info("Omitting instance level {} since it can be realized with a higher level number of instances ({}) without pruning attributes.", numInstances, maxLevelThatCanBeSMOTEDWithoutPruningAttributes);
			//				continue;
			//			}

			this.logger.info("Starting process for SMOTE file generation to cope with {} instances. The number of attributes managable for this limit is {}. The original dataset has {} attribute. Pruning {} attributes.", numInstances, numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment, numAttributesProvided, numAttributesProvided - numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment);

			if (numInstances <= origCompleteDataset.size() && numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment <= origCompleteDataset.getNumAttributes()) {
				this.logger.debug("No SMOTE file required for {} instances and {} attributes, because the original dataset already contains {} instances and {} attributes.", numInstances, numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment, origCompleteDataset.size(), origCompleteDataset.getNumAttributes());
				continue;
			}
			this.logger.debug("Original dataset has {} instances and {} attributes, but we require {} instances and {} attributes for the next level. Producing a SMOTE file for this level.", origCompleteDataset.size(), origCompleteDataset.getNumAttributes(), numInstances, numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment);

			/* if the SMOTE file already exists, ignore it */
			if (this.doesExtensionExist(id, numInstances)) {
				this.logger.debug("SMOTE extension for size {} already exists, skipping.", numInstances);
				continue;
			}

			/* reading in the dataset with the respective number of columns */
			this.logger.debug("Setting the number of effectively treated attributes (considering the existent {} attributes in the dataset to {})", numAttributesProvided, numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment);
			ILabeledDataset<?> origDS = origCompleteDataset;
			if (numAttributesProvided > numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment) {
				List<Integer> ignoredColumns = this.sortColumnsByRelevance(origCompleteDataset, true).stream().limit(numAttributesProvided - numAttributesThereticallyAllowedForAnyNumberOfInstancesInSegment).collect(Collectors.toList());
				this.logger.info("Reading in original dataset with reduced number of columns: {}.", origCompleteDataset.getNumAttributes() - ignoredColumns.size());
				origDS = OpenMLDatasetReader.deserializeClassificationDataset(id, ignoredColumns);
			}
			this.logger.debug("Original dataset from which SMOTE extension is drawn has format {} x {}", origDS.size(), origDS.getNumAttributes());

			/* get SMOTE extension  */
			WekaInstances ds = new WekaInstances((ILabeledDataset<?>)origDS.createCopy());
			if (numInstances > origDS.size()) {
				ds.addAll(this.createSMOTEExtensionOfDataset(origDS, numInstances - origDS.size()));
			}
			else {
				this.logger.debug("Dataset already large enough in terms of instances. Possibly removing some.");
			}
			this.logger.debug("Possibly reducing, in a stratified manner, the dataset.");
			ds = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, 0, (5 + numInstances) * 1.0 / ds.size()).get(0);
			this.logger.info("Produced SMOTE extension of size {} x {}. Now potentially generating new columns.", ds.size(), ds.getNumAttributes());
			// Now checking for duplicates.", ds.size());
			// int duplicates = 0;
			// int n = ds.size();
			// for (int i = 0; i < n; i++) {
			// ILabeledInstance i1 = ds.get(i);
			// for (int j = 0; j < i; j ++) {
			// ILabeledInstance i2 = ds.get(j);
			// boolean foundDeviation = false;
			// for (int k = 0; k < numAttributes; k++) {
			// if (!i1.getAttributeValue(k).equals(i2.getAttributeValue(k))) {
			// foundDeviation = true;
			// break;
			// }
			// }
			// if (!foundDeviation) {
			// duplicates++;
			// }
			// }
			// }
			// if (duplicates > 0) {
			// throw new IllegalStateException("Created " + duplicates + " duplicates. Duplicates are forbidden though!!");
			// }

			/* Append attributes */
			int maxAttributes = getMaximumAttributesManagableForNumberOfInstances(numInstances);
			int additionalAttributes = Math.max(0, maxAttributes - ds.getNumAttributes());
			this.logger.debug("Now generating {} artificial columns.", additionalAttributes);
			List<IAttribute> attributes = new ArrayList<>(origCompleteDataset.getListOfAttributes());
			for (int j = 0; j < additionalAttributes; j++) {
				attributes.add(new NumericAttribute("artf_" + j));
			}
			LabeledInstanceSchema extendedScheme = new LabeledInstanceSchema(origCompleteDataset.getRelationName(), attributes, origCompleteDataset.getLabelAttribute());
			this.logger.debug("New scheme created, now adding the instances.");
			Dataset expandedDatataset = new Dataset(extendedScheme);
			int m = ds.size();
			int count = 0;
			double nextStep = 0.05;
			Random random = new Random(origCompleteDataset.getRelationName().hashCode());
			for (ILabeledInstance inst : ds) {
				count++;
				List<Object> values = new ArrayList<>(Arrays.asList(inst.getAttributes()));
				for (int j = 0; j < additionalAttributes; j++) {
					values.add(random.nextDouble());
				}
				DenseInstance newInst = new DenseInstance(values, inst.getLabel());
				expandedDatataset.add(newInst);
				if (count * 1.0 / m >= nextStep) {
					this.logger.debug("Progress: {}%", MathExt.round(count * 100.0 / m, 2));
					nextStep += 0.05;
				}
				this.logger.trace("{}/{}", count, m);
			}
			this.logger.info("Created expanded dataset of size {} x {}", expandedDatataset.size(), expandedDatataset.getNumAttributes());

			/* write SMOTE extension */
			File file = this.getFileForSMOTEExtension(id, numInstances);
			this.logger.info("Writing SMOTE extension (including possibly artificial columns) to file {}", file);
			ArffDatasetAdapter.serializeDataset(file, expandedDatataset);
			this.logger.info("Serialization finished. Now checking it.");

			/* check the SMOTE extension */
			ILabeledDataset<?> recoveredSMOTEFile = this.getDatasetExtension(id, numInstances, Arrays.asList());
			if (recoveredSMOTEFile.size() + origCompleteDataset.size() < numInstances) {
				throw new IllegalStateException("The extension has not enough instances!");
			}
			if (recoveredSMOTEFile.getNumAttributes() != expandedDatataset.getNumAttributes()) {
				throw new IllegalStateException("The number of attributes in the extension file does not match! Expected to see " + origDS.getNumAttributes() + " but was " + recoveredSMOTEFile.getNumAttributes());
			}
		}
	}

	private WekaInstances getDatasetExtension(final int openmlid, final int maxNumberOfInstances, final List<Integer> ignoreColumns) throws DatasetDeserializationFailedException {
		this.logger.info("Loading SMOTE extension for openmlid {} with at most {} instances and ignoring {} attributes.", openmlid, maxNumberOfInstances, ignoreColumns.size());
		ILabeledDataset<?> extension = ArffDatasetAdapter.readDataset(false, this.getFileForSMOTEExtension(openmlid, maxNumberOfInstances), -1, maxNumberOfInstances, ignoreColumns);
		this.logger.info("Loaded SMOTE extension from file. Size is {} x {}", extension.size(), extension.getNumAttributes());
		return new WekaInstances(extension);
	}

	public WekaInstances createSMOTEExtensionOfDataset(final ILabeledDataset<?> originalDataset, final int sizeOfExtension) throws Exception {
		if (sizeOfExtension <= 0) {
			throw new IllegalArgumentException("It does not make sense to produce a negative SMOTE extension!");
		}
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

		/* running SMOTE */
		Instances upsampled = new Instances(inst);
		Instances extension = new Instances(inst);
		extension.clear();
		this.logger.info("Now producing a dataset of {} instances with SMOTE.", sizeOfExtension);
		List<String> classValues = new ArrayList<>(WekaUtil.getClassesActuallyContainedInDataset(inst));
		if (this.logger.isDebugEnabled()) {
			IntStream.range(0, classValues.size())
			.forEach(cIndex -> this.logger.debug("Number of occurrences of class {} ({}) is {} ({}%)", cIndex, inst.classAttribute().value(cIndex),
					inst.stream().filter(linst -> WekaUtil.getClassName(linst).equals(classValues.get(cIndex))).count(),
					MathExt.round(inst.stream().filter(linst -> WekaUtil.getClassName(linst).equals(classValues.get(cIndex))).count() * 100.0 / inst.size(), 2)));
		}
		int numClassess = classValues.size();
		while (extension.size() < sizeOfExtension) {
			int numberOfMissingInstances = 10 * numClassess + sizeOfExtension - extension.size(); // 10 is just a buffer to avoid stupid cases of 2 more required instances
			this.logger.info("Next SMOTE iteration round. Available instances in extension: {}. Missing instances: {}", extension.size(), numberOfMissingInstances);

			/* in each round, consider each class as the minority class once */
			double percentage = (numberOfMissingInstances * 100.0 / inst.size());
			// percentage *= 2; // this is because the algorithm does not really respect the required number.
			double percentageForEachClass = percentage;
			this.logger.info("Setting percentage to {}, which is {} for each of the {} classes.", percentage, percentageForEachClass, numClassess);

			List<Instance> addedInstances = new ArrayList<>();
			int i = 0;
			for (int c = 0; c < classValues.size(); c++) { // the class value is not used because there is a bug in SMOTE that does not allow to do so!
				SMOTE smote = new SMOTE();
				smote.setPercentage(percentageForEachClass);
				smote.setInputFormat(inst);
				smote.setClassValue((c + 1) + "");
				this.logger.debug("Starting SMOTE using {} instances. Applying for class {}: {} ({}/{}).", upsampled.size(), c, smote.getClassValue(), ++i, numClassess);
				List<Instance> instancesBefore = new ArrayList<>(upsampled);
				List<Instance> locallUpsampled = null;
				while (locallUpsampled == null) {
					try {
						locallUpsampled = Filter.useFilter(upsampled, smote);
					} catch (NullPointerException e) {
						System.err.println("NPE. Rertrying.");
					}
				}
				this.logger.debug("SMOTE finished. Number of instances after upsampling is now {}", locallUpsampled.size());
				for (int j = instancesBefore.size(); j < locallUpsampled.size(); j++) {
					addedInstances.add(locallUpsampled.get(j));
				}
				this.logger.debug("Number of totally added instances up to now in this SMOTE round is {}", addedInstances.size());
				if (this.logger.isDebugEnabled()) {
					IntStream.range(0, classValues.size())
					.forEach(cIndex -> this.logger.debug("Number of occurrences of class {} ({}) is {} ({}%)", cIndex, inst.classAttribute().value(cIndex),
							addedInstances.stream().filter(linst -> WekaUtil.getClassName(linst).equals(classValues.get(cIndex))).count(),
							MathExt.round(addedInstances.stream().filter(linst -> WekaUtil.getClassName(linst).equals(classValues.get(cIndex))).count() * 100.0 / addedInstances.size(), 2)));
				}
			}
			extension.addAll(addedInstances);
			upsampled.addAll(addedInstances);
			this.logger.info("Finished SMOTE round. Extension has now {} instances. Overall upsampled dataset has now {} instances.", extension.size(), upsampled.size());
		}
		this.logger.info("Finished SMOTE sampling.");
		if (this.logger.isDebugEnabled()) {
			IntStream.range(0, classValues.size())
			.forEach(cIndex -> this.logger.debug("Number of occurrences of class {} ({}) is {} ({}%)", cIndex, inst.classAttribute().value(cIndex),
					extension.stream().filter(linst -> WekaUtil.getClassName(linst).equals(classValues.get(cIndex))).count(),
					MathExt.round(extension.stream().filter(linst -> WekaUtil.getClassName(linst).equals(classValues.get(cIndex))).count() * 100.0 / extension.size(), 2)));
		}
		return new WekaInstances(extension);
	}

	public List<ILabeledDataset<?>> createSizeAdjustedTrainTestSplit(final int openmlid, final int trainsize, final int minTestSize, final int numAttributes, final Random random) throws Exception {
		this.logger.info("Reading in dataset {}", openmlid);

		/* load original dataset and then load the SMOTE file if necessary */
		ILabeledDataset<?> dsOrig = OpenMLDatasetReader.deserializeDataset(openmlid, Arrays.asList());
		if (dsOrig.size() < trainsize + minTestSize && !this.doesExtensionExist(openmlid, trainsize + minTestSize)) {
			throw new IllegalArgumentException("Dataset has not sufficient points (" + dsOrig.size() + " instead of the required " + (trainsize + minTestSize) + "), but no SMOTE file existent for openmlid " + openmlid);
		}

		List<Integer> ignoredColumns = this.getIgnoredColumnsForDatasetAndGivenNumberOfMaxRemainingAttributes(dsOrig, trainsize + minTestSize, numAttributes);

		this.logger.debug("Ignoring {} columns and including {} columns. Enable trace to see which ones are ignored.", ignoredColumns.size(), numAttributes);
		if (this.logger.isTraceEnabled()) {
			ignoredColumns.forEach(c -> this.logger.trace("Ignoring column {}", c));
		}

		/* get reduced datasets  */
		WekaInstances extendedDataset = this.getDatasetExtension(openmlid, trainsize + minTestSize, ignoredColumns);
		if (extendedDataset.getNumAttributes() != numAttributes) {
			throw new IllegalStateException("Reduced extension dataset has " + extendedDataset.getNumAttributes() + " attributes but should be " + numAttributes + ".");
		}
		this.logger.info("Computed adjusted dataset extension of size {} x {}. Now computing a split with {} training examples and (at least) {} test instances.", extendedDataset.size(), extendedDataset.getNumAttributes(), trainsize, minTestSize);
		return this.createSizeAdjustedTrainTestSplit(extendedDataset, trainsize, minTestSize, numAttributes, random);
	}

	private List<ILabeledDataset<?>> createSizeAdjustedTrainTestSplit(final ILabeledDataset<?> dataset, final int trainsize, final int minTestSize, final int numAttributes,
			final Random random) throws Exception {

		if (!(dataset instanceof WekaInstances)) {
			throw new IllegalArgumentException("Dataset must be of type " + WekaInstances.class + " but is of type " + dataset.getClass());
		}
		/* first make sure that the dataset size is at least the required train size + the min test size. If necessary, apply SMOTE to get more instances */
		int minimumRequiredInstances = trainsize + minTestSize;
		this.logger.info("Creating a split for {} = {} + {} data points. Number of existing points in data: {}.", minimumRequiredInstances, trainsize, minTestSize, dataset.size());
		if (dataset.size()  < trainsize + minTestSize) {
			throw new IllegalArgumentException("The given dataset is too small to create a split for " + (trainsize + minTestSize) + " instances.");
		}

		/* this method is not responsible for reducing the number of attributes. This should happen before. */
		if (dataset.getNumAttributes() != numAttributes) {
			throw new IllegalArgumentException(
					"Observing " + dataset.getNumAttributes() + " instead of the required " + numAttributes + " attributes. Please make sure that obsolete attributes are removed already when reading the dataset.");
		}

		/* if the dataset is large enough to provide all necessary instances for the training set, use the complementary data only for testing */
		this.logger.debug("Possibly shrinking dataset for speedup.");
		while (dataset.size() > trainsize + minTestSize + 5000) {
			dataset.remove(trainsize + minTestSize + 5000);
		}
		this.logger.debug("Dataset size prior to split is {} x {}", dataset.size(), dataset.getNumAttributes());
		List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, random, 1.0 * trainsize / dataset.size());
		ILabeledDataset<?> testFold = split.get(1);
		this.logger.debug("Finished split. Test fold has {} instances, possibly erazing items.", testFold.size());
		while (testFold.size() > minTestSize) {
			testFold.remove(minTestSize);
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
