package tpami.basealgorithmlearning.datagathering;

import java.io.File;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.WekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import tpami.basealgorithmlearning.datagathering.classification.defaultparams.DefaultBaseLearnerConfigContainer;
import tpami.basealgorithmlearning.datagathering.classification.defaultparams.IDefaultBaseLearnerExperimentConfig;
import tpami.basealgorithmlearning.regression.BasicDatasetFeatureGenerator;
import tpami.basealgorithmlearning.regression.DatasetVarianceFeatureGenerator;
import weka.classifiers.rules.JRip;

public class MetaFeatureComputer {

	private final IDatabaseAdapter adapter;
	private final Map<Integer, Integer> numInstances = new HashMap<>();

	private final String tablename;
	private BasicDatasetFeatureGenerator basicFeatureGen = new BasicDatasetFeatureGenerator();
	private DatasetVarianceFeatureGenerator varFeatureGen = new DatasetVarianceFeatureGenerator();
	private ILabeledDataset<?> lastDataset;
	private int lastDatasetId;

	public static void main(final String[] args) throws Exception {
		MetaFeatureComputer c = new MetaFeatureComputer("dataset_mf1", args[0]);
		c.getIDs();
		//		c.createTable();
		//		c.initializeTableFromConfig();
		//		c.fillMissingValues(Integer.parseInt(args[1]));
	}

	public MetaFeatureComputer(final String tablename, final String configfile) {
		IDatabaseConfig config = (IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File(configfile));
		//		IRestDatabaseConfig config = (IRestDatabaseConfig) ConfigFactory.create(IRestDatabaseConfig.class).loadPropertiesFromFile(new File(configfile));
		this.adapter = new SQLAdapter(config);
		this.adapter.setLoggerName("example");
		this.tablename = tablename;

		/* set up number of instances */
		this.numInstances.put(3, 3196);
		this.numInstances.put(6, 20000);
		this.numInstances.put(12, 2000);
		this.numInstances.put(14, 2000);
		this.numInstances.put(16, 2000);
		this.numInstances.put(18, 2000);
		this.numInstances.put(21, 1728);
		this.numInstances.put(22, 2000);
		this.numInstances.put(23, 1473);
		this.numInstances.put(24, 8124);
		this.numInstances.put(26, 12960);
		this.numInstances.put(28, 5620);
		this.numInstances.put(30, 5473);
		this.numInstances.put(31, 1000);
		this.numInstances.put(32, 10992);
		this.numInstances.put(36, 2310);
		this.numInstances.put(38, 3772);
		this.numInstances.put(44, 4601);
		this.numInstances.put(46, 3190);
		this.numInstances.put(57, 3772);
		this.numInstances.put(60, 5000);
		this.numInstances.put(179, 48842);
		this.numInstances.put(180, 110393);
		this.numInstances.put(181, 1484);
		this.numInstances.put(182, 6430);
		this.numInstances.put(183, 4177);
		this.numInstances.put(184, 28056);
		this.numInstances.put(185, 1340);
		this.numInstances.put(273, 120919);
		this.numInstances.put(293, 581012);
		this.numInstances.put(300, 7797);
		this.numInstances.put(351, 488565);
		this.numInstances.put(354, 1025010);
		this.numInstances.put(357, 98528);
		this.numInstances.put(389, 2463);
		this.numInstances.put(390, 9558);
		this.numInstances.put(391, 1504);
		this.numInstances.put(392, 1003);
		this.numInstances.put(393, 3075);
		this.numInstances.put(395, 1657);
		this.numInstances.put(396, 3204);
		this.numInstances.put(398, 1560);
		this.numInstances.put(399, 11162);
		this.numInstances.put(401, 1050);
		this.numInstances.put(554, 70000);
		this.numInstances.put(679, 1024);
		this.numInstances.put(715, 1000);
		this.numInstances.put(718, 1000);
		this.numInstances.put(720, 4177);
		this.numInstances.put(722, 15000);
		this.numInstances.put(723, 1000);
		this.numInstances.put(727, 40768);
		this.numInstances.put(728, 4052);
		this.numInstances.put(734, 13750);
		this.numInstances.put(735, 8192);
		this.numInstances.put(737, 3107);
		this.numInstances.put(740, 1000);
		this.numInstances.put(741, 1024);
		this.numInstances.put(743, 1000);
		this.numInstances.put(751, 1000);
		this.numInstances.put(752, 8192);
		this.numInstances.put(761, 8192);
		this.numInstances.put(772, 2178);
		this.numInstances.put(797, 1000);
		this.numInstances.put(799, 1000);
		this.numInstances.put(803, 7129);
		this.numInstances.put(806, 1000);
		this.numInstances.put(807, 8192);
		this.numInstances.put(813, 1000);
		this.numInstances.put(816, 8192);
		this.numInstances.put(819, 9517);
		this.numInstances.put(821, 22784);
		this.numInstances.put(822, 20640);
		this.numInstances.put(823, 20640);
		this.numInstances.put(833, 8192);
		this.numInstances.put(837, 1000);
		this.numInstances.put(843, 22784);
		this.numInstances.put(845, 1000);
		this.numInstances.put(846, 16599);
		this.numInstances.put(847, 6574);
		this.numInstances.put(849, 1000);
		this.numInstances.put(866, 1000);
		this.numInstances.put(871, 3848);
		this.numInstances.put(881, 40768);
		this.numInstances.put(897, 1161);
		this.numInstances.put(901, 40768);
		this.numInstances.put(903, 1000);
		this.numInstances.put(904, 1000);
		this.numInstances.put(910, 1000);
		this.numInstances.put(912, 1000);
		this.numInstances.put(913, 1000);
		this.numInstances.put(914, 2001);
		this.numInstances.put(917, 1000);
		this.numInstances.put(923, 8641);
		this.numInstances.put(930, 1302);
		this.numInstances.put(934, 1156);
		this.numInstances.put(953, 3190);
		this.numInstances.put(958, 2310);
		this.numInstances.put(959, 12960);
		this.numInstances.put(962, 2000);
		this.numInstances.put(966, 1340);
		this.numInstances.put(971, 2000);
		this.numInstances.put(976, 9961);
		this.numInstances.put(977, 20000);
		this.numInstances.put(978, 2000);
		this.numInstances.put(979, 5000);
		this.numInstances.put(980, 5620);
		this.numInstances.put(991, 1728);
		this.numInstances.put(993, 7019);
		this.numInstances.put(995, 2000);
		this.numInstances.put(1000, 3772);
		this.numInstances.put(1002, 7485);
		this.numInstances.put(1018, 8844);
		this.numInstances.put(1019, 10992);
		this.numInstances.put(1020, 2000);
		this.numInstances.put(1021, 5473);
		this.numInstances.put(1036, 14395);
		this.numInstances.put(1037, 4562);
		this.numInstances.put(1039, 4229);
		this.numInstances.put(1040, 14395);
		this.numInstances.put(1041, 3468);
		this.numInstances.put(1042, 3468);
		this.numInstances.put(1049, 1458);
		this.numInstances.put(1050, 1563);
		this.numInstances.put(1053, 10885);
		this.numInstances.put(1059, 121);
		this.numInstances.put(1067, 2109);
		this.numInstances.put(1068, 1109);
		this.numInstances.put(1069, 5589);
		this.numInstances.put(1111, 50000);
		this.numInstances.put(1112, 50000);
		this.numInstances.put(1114, 50000);
		this.numInstances.put(1116, 6598);
		this.numInstances.put(1119, 32561);
		this.numInstances.put(1120, 19020);
		this.numInstances.put(1128, 1545);
		this.numInstances.put(1130, 1545);
		this.numInstances.put(1134, 1545);
		this.numInstances.put(1138, 1545);
		this.numInstances.put(1139, 1545);
		this.numInstances.put(1142, 1545);
		this.numInstances.put(1146, 1545);
		this.numInstances.put(1161, 1545);
		this.numInstances.put(1166, 1545);
		this.numInstances.put(1216, 1496391);
		this.numInstances.put(1242, 98528);
		this.numInstances.put(1457, 1500);
		this.numInstances.put(1485, 2600);
		this.numInstances.put(1486, 34465);
		this.numInstances.put(1501, 1593);
		this.numInstances.put(1569, 1025000);
		this.numInstances.put(4136, 600);
		this.numInstances.put(4137, 1150);
		this.numInstances.put(4541, 101766);
		this.numInstances.put(4552, 5665);
		this.numInstances.put(23380, 2796);
		this.numInstances.put(23512, 98050);
		this.numInstances.put(40497, 3772);
		this.numInstances.put(40594, 2000);
		this.numInstances.put(40685, 58000);
		this.numInstances.put(40691, 1599);
		this.numInstances.put(40900, 5100);
		this.numInstances.put(40926, 20000);
		this.numInstances.put(40927, 60000);
		this.numInstances.put(40971, 1000);
		this.numInstances.put(40975, 1728);
		this.numInstances.put(41026, 7000);
		this.numInstances.put(41064, 58000);
		this.numInstances.put(41065, 62000);
		this.numInstances.put(41066, 1567);
		this.numInstances.put(41143, 2984);
		this.numInstances.put(41146, 5124);
		this.numInstances.put(41164, 8237);
		this.numInstances.put(41946, 3772);
		this.numInstances.put(41991, 270912);
	}

	public void getIDs() throws Exception {
		int[] ids = {3, 6, 12, 14, 16, 18, 21, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 44, 46, 57, 60, 179, 180, 181, 182, 183, 184, 185, 273, 293, 300, 351, 354, 357, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1037, 1039, 1040, 1041, 1042, 1049, 1050, 1053, 1059, 1067, 1068, 1069, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1242, 1457, 1485, 1486, 1501, 1569, 4136, 4137, 4541, 4552, 23380, 23512, 40497, 40685, 40691, 40900, 40926, 40927, 40971, 40975, 41026, 41064, 41065, 41066, 41143, 41146, 41164, 41946, 41991};
		for (int id : ids) {

			if (id == 993) {
				ILabeledDataset<?> ds = OpenMLDatasetReader.deserializeDataset(id);
				WekaInstances wi = new WekaInstances(ds);
				int instancesForId = wi.size();
				if (this.numInstances.get(id) != instancesForId) {
					throw new IllegalStateException(id + " should have " + instancesForId + " instances, but db says " + this.numInstances.get(id));
				}

				List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, 7, 100.0 / ds.size());
				WekaClassifier rt = new WekaClassifier(new JRip());
				System.out.println("Training on " + id);
				rt.fit(split.get(0));
				System.out.println("Ready now testing.");
				for (ILabeledInstance i : split.get(1)) {
					rt.predict(i);
				}
			}
		}
	}

	public void rewriteEntries() throws SQLException {
		System.out.println("Setting logger");
		this.adapter.setLoggerName("example");
		System.out.println("Posing query");
		Collection<IKVStore> rows = this.adapter
				.getResultsOfQuery("SELECT t2.eval_id, t1.openmlid, t1.datapoints, t1.seed FROM `dataset_metafeatures` as t2 JOIN evaluations_classifiers_decisiontable as t1 on t1.experiment_id = t2.eval_id WHERE 1");
		System.out.println("Ready. Going over rows.");
		for (IKVStore row : rows) {
			Map<String, Object> vals = new HashMap<>();
			vals.put("traindatasize", row.get("datapoints"));
			vals.put("seed", row.get("seed"));
			Map<String, Object> cond = new HashMap<>();
			cond.put("eval_id", row.get("eval_id"));
			this.adapter.update("dataset_metafeatures", vals, cond);
			System.out.println(row);
		}
	}

	public void initializeTableFromConfig() throws Exception {

		/* read in existing entries of both tables */
		// Collection<IKVStore> rows = this.adapter.getRowsOfTable(table);

		/* first check possible combinations of base learner configs */
		DefaultBaseLearnerConfigContainer blcc = new DefaultBaseLearnerConfigContainer("conf/dbcon-local.conf", "weka.classifiers.trees.RandomTree");
		List<List<? extends Object>> trainCombos = new ArrayList<>(SetUtil.cartesianProduct(Arrays.asList(blcc.getConfig().openMLIDs(), blcc.getConfig().datapoints(), blcc.getConfig().seeds())));

		/* derive test combos (as total sizes - train sizes) resulting from this */
		List<List<? extends Object>> extendedCombos = trainCombos.stream().map(l -> {
			List<Object> l2 = new ArrayList<>(l);
			if (!this.numInstances.containsKey((int) l.get(0))) {
				throw new IllegalStateException("No number of instances available for dataset " + l.get(0));
			}
			int numTestInstances = this.numInstances.get((int) l.get(0)) - (int) l.get(1);
			if (numTestInstances <= 0) {
				return null;
			}
			l2.add(2, numTestInstances);
			return l2;
		}).filter(Objects::nonNull).collect(Collectors.toList());
		this.adapter.insertMultiple(this.tablename, Arrays.asList("openmlid", "datapoints_fold1", "datapoints_fold2", "seed"), extendedCombos);

	}

	public void fillTableFromRegressionTable(final String table, final String regressionTable) throws SQLException {

		/* read in existing entries */
		Collection<IKVStore> rows = this.adapter.getRowsOfTable(table);
		for (IKVStore row : rows) {
			System.out.println(row);
		}
	}

	public void createTable() throws SQLException {
		IDefaultBaseLearnerExperimentConfig cfg = (IDefaultBaseLearnerExperimentConfig) ConfigFactory.create(IDefaultBaseLearnerExperimentConfig.class).loadPropertiesFromFile(new File("conf/experiments/defaultparams/preprocessor.conf"));
		List<String> fieldDescriptors = cfg.getResultFields().stream().filter(n -> n.contains("before")).collect(Collectors.toList());
		List<String> fieldNames = new ArrayList<>();
		Map<String, String> fieldTypes = new HashMap<>();
		fieldTypes.put("dsv_id", "int(8)");
		fieldNames.add("openmlid");
		fieldTypes.put("openmlid", "int(6)");
		fieldNames.add("datapoints_fold1");
		fieldTypes.put("datapoints_fold1", "int(8)");
		fieldNames.add("datapoints_fold2");
		fieldTypes.put("datapoints_fold2", "int(8)");
		fieldNames.add("attributes");
		fieldTypes.put("attributes", "int(6) NULL");
		fieldNames.add("seed");
		fieldTypes.put("seed", "int(2)");
		for (int f = 1; f <= 2; f++) {
			final int ff = f;
			fieldDescriptors.forEach(desc -> {
				String[] parts = desc.split(":");
				String name = parts[0];
				name = name.substring(0, name.length() - "_before".length());
				fieldNames.add("f" + ff + "_" + name);
				fieldTypes.put("f" + ff + "_" + name, parts[1] + " NULL");
			});
		}
		this.adapter.createTable(this.tablename, "dsv_id", fieldNames, fieldTypes, Arrays.asList());
	}

	public void fillMissingValues(final int parallelization) throws Exception {

		this.adapter.setLoggerName("example");
		List<IKVStore> rows = this.adapter
				.getResultsOfQuery("SELECT `dsv_id`, `openmlid`, `datapoints_fold1`, `datapoints_fold2`, `seed` FROM `" + this.tablename + "` WHERE f1_numinstances IS NULL ORDER BY openmlid");
		ExecutorService pool = Executors.newFixedThreadPool(parallelization);
		rows.forEach(r -> pool.submit(new MFComputer(r)));
		pool.shutdown();
	}

	private class MFComputer implements Runnable {

		private final IKVStore row;

		public MFComputer(final IKVStore row) {
			super();
			this.row = row;
		}

		@Override
		public void run() {
			try {
				Map<String, Object> condMap = new HashMap<>();
				condMap.put("dsv_id", this.row.getAsInt("dsv_id"));
				Map<String, Object> map = new HashMap<>();
				int openmlid = this.row.getAsInt("openmlid");
				int datapoints = this.row.getAsInt("datapoints_fold1");
				int seed = this.row.getAsInt("seed");
				System.out.println("Treating " + openmlid + " with " + datapoints + " datapoints on seed " + seed);
				ILabeledDataset<?> ds = (openmlid == MetaFeatureComputer.this.lastDatasetId) ? MetaFeatureComputer.this.lastDataset : OpenMLDatasetReader.deserializeDataset(openmlid);
				MetaFeatureComputer.this.lastDataset = ds;
				MetaFeatureComputer.this.lastDatasetId = openmlid;
				double portion = datapoints * 1.0 / ds.size();
				if (portion >= 1) {
					System.err.println("Canceling information, because number of datapoints is too high!");
					return;
				}
				System.out.println("Start to compute stratified split.");
				List<ILabeledDataset<?>> split;
				try {
					split = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, portion);
				}
				catch (Throwable e) {
					e.printStackTrace();
					throw e;
				}
				System.out.println("Done. Computing features of first split.");
				MetaFeatureComputer.this.basicFeatureGen.setPrefix("f1_");
				MetaFeatureComputer.this.varFeatureGen.setPrefix("f1_");
				map.putAll(MetaFeatureComputer.this.basicFeatureGen.getFeatureRepresentation(split.get(0)));
				map.putAll(MetaFeatureComputer.this.varFeatureGen.getFeatureRepresentation(split.get(0)));
				System.out.println("Done. Computing features of second split.");
				MetaFeatureComputer.this.basicFeatureGen.setPrefix("f2_");
				MetaFeatureComputer.this.varFeatureGen.setPrefix("f2_");
				map.putAll(MetaFeatureComputer.this.basicFeatureGen.getFeatureRepresentation(split.get(1)));
				map.putAll(MetaFeatureComputer.this.varFeatureGen.getFeatureRepresentation(split.get(1)));
				System.out.println("All features computed. Running update: " + map);
				MetaFeatureComputer.this.adapter.update(MetaFeatureComputer.this.tablename, map, condMap);
				System.out.println("Update completed.");
			} catch (Throwable e) {
				e.printStackTrace();
			}
		}
	}
}
