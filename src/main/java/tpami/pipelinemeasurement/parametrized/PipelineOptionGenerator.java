package tpami.pipelinemeasurement.parametrized;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.jaicore.basic.MathExt;
import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.experiments.IExperimentKeyGenerator;
import ai.libs.jaicore.logging.LoggerUtil;
import ai.libs.jaicore.ml.weka.WekaUtil;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.classification.pipeline.MLPipeline;
import tpami.basealgorithmlearning.EmptyOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.ANNOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.BNOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.DTOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.IBkOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.J48OptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.JRipOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.LMTOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.LogisticOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.NBOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.OneROptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.PartOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.REPOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.RFOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.RTOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.SLOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.SMOOptionGenerator;
import tpami.basealgorithmlearning.datagathering.classification.parametrized.optiongenerators.VPOptionGenerator;
import tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators.AdaBoostM1OptionGenerator;
import tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators.BaggingOptionGenerator;
import tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators.LogitBoostOptionGenerator;
import tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators.RandomCommitteeOptionGenerator;
import tpami.basealgorithmlearning.datagathering.metaalgorithm.parametrized.optiongenerators.RandomSubspaceOptionGenerator;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators.BestFirstOptionGenerator;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators.CFSSubsetOptionGenerator;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators.InfoGainAEOptionGenerator;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators.PCAOptionGenerator;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators.RankerOptionGenerator;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators.ReliefFOptionGenerator;
import tpami.basealgorithmlearning.datagathering.preprocessing.parametrized.optiongenerators.SymmetricalAttributeOptionGenerator;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.meta.RandomCommittee;
import weka.classifiers.meta.RandomSubSpace;
import weka.core.Randomizable;

public class PipelineOptionGenerator implements IExperimentKeyGenerator<String> {

	private static List<String> COMBOS = new ArrayList<>();
	private static Logger logger = LoggerFactory.getLogger(LoggerUtil.LOGGER_NAME_TESTEDALGORITHM);

	static {
		//		gen();
	}

	private static void gen() {
		List<String> baselearners = WekaUtil.getBasicLearners().stream().sorted((c1, c2) -> c1.substring(c1.lastIndexOf(".")).compareTo(c2.substring(c2.lastIndexOf(".")))).collect(Collectors.toList());
		baselearners.removeIf(baseLearner -> baseLearner.contains("M5") || baseLearner.contains("LinearRegression"));
		Collection<List<String>> combosPP = WekaUtil.getAdmissibleSearcherEvaluatorCombinationsForAttributeSelection();
		List<String> metaLearners = new ArrayList<>(WekaUtil.getMetaLearners());
		metaLearners.removeIf(c -> c.contains("AttributeSelectedClassifier") || c.contains("Vote") || c.contains("Stacking") || c.contains("MultiClassClassifier") || c.contains("Regression"));
		List<Integer> attRange = Arrays.asList(10, 50, 100, 200, 1000);
		List<Integer> instRange = Arrays.asList(100, 1000, 5000, 10000, 20000, 100000);
		System.out.println(metaLearners);
		int numPipelines = 0;
		List<Integer> openmlids = Arrays.asList(3, 6, 12, 14, 16, 18, 21, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 44, 46, 57, 60, 179, 180, 181, 182, 183, 184, 185, 273, 293, 300, 351, 354, 357, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1037, 1039, 1040, 1041, 1042, 1049, 1050, 1053, 1059, 1067, 1068, 1069, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1242, 1457, 1485, 1486, 1501, 1569, 4136, 4137, 4541, 4552, 23380, 23512, 40497, 40685, 40691, 40900, 40926, 40927, 40971, 40975, 41026, 41064, 41065, 41066, 41143, 41146, 41164, 41946, 41991);

		int pipelinesPerBaseLearner = 5;
		int pipelinesPerPreprocessor = 5;

		int totalNumberOfExperiments = pipelinesPerBaseLearner * pipelinesPerPreprocessor * baselearners.size() * combosPP.size() * attRange.size() * instRange.size();

		try {

			Random random = new Random(0);
			for (int numAttributes : attRange) {

				for (int numInstances : instRange) {

					if (numAttributes * numInstances > 300000000) {
						continue;
					}

					for (String baseLearner: baselearners) {

						for (int i = 0; i < pipelinesPerBaseLearner; i ++) {

							for (List<String> preProcessor: combosPP) {

								for (int j = 0; j < pipelinesPerPreprocessor; j++) {

									/* draw options for the base learner */
									String searchName = preProcessor.get(0);
									String evalName = preProcessor.get(1);
									IExperimentKeyGenerator<String> optionGeneratorBaseLearner = getOptionGenerator(baseLearner);
									IExperimentKeyGenerator<String> optionGeneratorSearch = getOptionGenerator(searchName);
									IExperimentKeyGenerator<String> optionGeneratorEvaluation = getOptionGenerator(evalName);
									boolean ok = false;
									while (!ok) {

										/* meta learners */
										int metaLearnerIndex = random.nextInt(metaLearners.size() + 1);
										String ml = null;
										if (metaLearnerIndex < metaLearners.size()) {
											ml = metaLearners.get(metaLearnerIndex);
										}

										try {

											Classifier c;
											Classifier baseClassifier = AbstractClassifier.forName(baseLearner, optionGeneratorBaseLearner.getNumberOfValues() > 0 ? optionGeneratorBaseLearner.getValue(random.nextInt(optionGeneratorBaseLearner.getNumberOfValues())).split(" ") : null);
											if (ml != null) {
												IExperimentKeyGenerator<String> optionGeneratorMetaLearner = getOptionGenerator(ml);
												SingleClassifierEnhancer metaClassifier = (SingleClassifierEnhancer)AbstractClassifier.forName(ml, optionGeneratorMetaLearner.getNumberOfValues() > 0 ? optionGeneratorMetaLearner.getValue(random.nextInt(optionGeneratorMetaLearner.getNumberOfValues())).split(" ") : null);
												if ((metaClassifier instanceof RandomSubSpace || metaClassifier instanceof RandomCommittee) && !(baseClassifier instanceof Randomizable)) {
													continue;
												}
												metaClassifier.setClassifier(baseClassifier);
												c = metaClassifier;
											}
											else {
												c = baseClassifier;
											}
											ASSearch search = ASSearch.forName(searchName, optionGeneratorSearch.getNumberOfValues() > 0 ? optionGeneratorSearch.getValue(random.nextInt(optionGeneratorSearch.getNumberOfValues())).split(" ") : null);
											ASEvaluation eval = ASEvaluation.forName(evalName, optionGeneratorEvaluation.getNumberOfValues() > 0 ? optionGeneratorEvaluation.getValue(random.nextInt(optionGeneratorEvaluation.getNumberOfValues())).split(" ") : null);
											WekaClassifier pipeline = new WekaClassifier(new MLPipeline(search,  eval, c));

											ObjectMapper om = new ObjectMapper();
											String str = om.writeValueAsString(pipeline.getConstructionPlan());

											int openmlid = SetUtil.getRandomElement(openmlids, random);

											COMBOS.add("{\"openmlid\": \"" + openmlid + "\", \"numinstances\": \"" + numInstances + "\", \"numattributes\": \"" + numAttributes + "\", \"pipeline\": " + str + "}");
											logger.debug("Added {}-{}-{} for openmlid {}", preProcessor, baseClassifier, metaLearnerIndex, openmlid);
											logger.info("Progress: {}%", MathExt.round(COMBOS.size() * 100.0 / totalNumberOfExperiments, 2));
											ok = true;
										}
										catch (Exception e) {
											logger.debug("Ignore combo {}", e.getMessage());
										}
									}
								}
							}
						}
					}
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(COMBOS.size());
	}

	public static IExperimentKeyGenerator<String> getOptionGenerator(final String baseLearner) {
		switch (baseLearner) {
		case "weka.classifiers.bayes.BayesNet":
			return new BNOptionGenerator();
		case "weka.classifiers.trees.DecisionStump":
		case "weka.classifiers.lazy.KStar":
		case "weka.classifiers.bayes.NaiveBayesMultinomial":
		case "weka.classifiers.rules.ZeroR":
			return new EmptyOptionGenerator(); // has no options
		case "weka.classifiers.rules.DecisionTable":
			return new DTOptionGenerator();
		case "weka.classifiers.lazy.IBk":
			return new IBkOptionGenerator();
		case "weka.classifiers.trees.J48":
			return new J48OptionGenerator();
		case "weka.classifiers.rules.JRip":
			return new JRipOptionGenerator();
		case "weka.classifiers.trees.LMT":
			return new LMTOptionGenerator();
		case "weka.classifiers.functions.Logistic":
			return new LogisticOptionGenerator();
		case "weka.classifiers.functions.MultilayerPerceptron":
			return new ANNOptionGenerator();
		case "weka.classifiers.bayes.NaiveBayes":
			return new NBOptionGenerator();
		case "weka.classifiers.rules.OneR":
			return new OneROptionGenerator();
		case "weka.classifiers.rules.PART":
			return new PartOptionGenerator();
		case "weka.classifiers.trees.REPTree":
			return new REPOptionGenerator();
		case "weka.classifiers.trees.RandomForest":
			return new RFOptionGenerator();
		case "weka.classifiers.trees.RandomTree":
			return new RTOptionGenerator();
		case "weka.classifiers.functions.SimpleLogistic":
			return new SLOptionGenerator();
		case "weka.classifiers.functions.SMO":
			return new SMOOptionGenerator();
		case "weka.classifiers.functions.VotedPerceptron":
			return new VPOptionGenerator();

			/* meta learner */
		case "weka.classifiers.meta.AdaBoostM1":
			return new AdaBoostM1OptionGenerator();
		case "weka.classifiers.meta.Bagging":
			return new BaggingOptionGenerator();
		case "weka.classifiers.meta.LogitBoost":
			return new LogitBoostOptionGenerator();
		case "weka.classifiers.meta.RandomCommittee":
			return new RandomCommitteeOptionGenerator();
		case "weka.classifiers.meta.RandomSubSpace":
			return new RandomSubspaceOptionGenerator();

			/* pre-processors */
		case "weka.attributeSelection.Ranker":
			return new RankerOptionGenerator();
		case "weka.attributeSelection.BestFirst":
			return new BestFirstOptionGenerator();
		case "weka.attributeSelection.GreedyStepwise":
			return new RankerOptionGenerator(); // same as for ranker
		case "weka.attributeSelection.CorrelationAttributeEval":
			return new EmptyOptionGenerator();
		case "weka.attributeSelection.CfsSubsetEval":
			return new CFSSubsetOptionGenerator();
		case "weka.attributeSelection.GainRatioAttributeEval":
			return new EmptyOptionGenerator();
		case "weka.attributeSelection.InfoGainAttributeEval":
			return new InfoGainAEOptionGenerator();
		case "weka.attributeSelection.OneRAttributeEval":
			return new OneROptionGenerator();
		case "weka.attributeSelection.PrincipalComponents":
			return new PCAOptionGenerator();
		case "weka.attributeSelection.ReliefFAttributeEval":
			return new ReliefFOptionGenerator();
		case "weka.attributeSelection.SymmetricalUncertAttributeEval":
			return new SymmetricalAttributeOptionGenerator();
		default:
			throw new IllegalArgumentException("No option generator for learner " + baseLearner);
		}
	}

	@Override
	public int getNumberOfValues() {
		return COMBOS.size();
	}

	@Override
	public String getValue(final int i) {
		return COMBOS.get(i);
	}

	@Override
	public boolean isValueValid(final String value) {
		return true;
	}
}
