package tpami.basealgorithmlearning.regression;

import java.io.File;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.sql.rest.IRestDatabaseConfig;
import ai.libs.jaicore.db.sql.rest.RestSqlAdapter;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import tpami.basealgorithmlearning.datagathering.classification.defaultparams.IDefaultBaseLearnerExperimentConfig;

/**
 * Turns a dataset with observations of a classifier into a regression dataset with the runtime information
 *
 * @author Felix Mohr
 */
class LogDatasetToRegressionDatasetConverter {

	private final IDatabaseAdapter adapter;
	private BasicDatasetFeatureGenerator basicFeatureGen = new BasicDatasetFeatureGenerator();
	private DatasetVarianceFeatureGenerator varFeatureGen = new DatasetVarianceFeatureGenerator();
	private ILabeledDataset<?> lastDataset;
	private int lastDatasetId;

	public static void main(final String[] args) throws Exception {
		LogDatasetToRegressionDatasetConverter con = new LogDatasetToRegressionDatasetConverter(new RestSqlAdapter((IRestDatabaseConfig)ConfigFactory.create(IRestDatabaseConfig.class).loadPropertiesFromFile(new File(args[0]))));
		String classifier = args[1];
		con.convertTable("evaluations_classifiers_" + classifier, "regression_classifiers_" + classifier);
		//		con.createRegressionTable("regression_classifiers_" + classifier);
	}

	public LogDatasetToRegressionDatasetConverter(final IDatabaseAdapter adapter) {
		this.adapter = adapter;
		adapter.setLoggerName("example");
	}

	public void createRegressionTable(final String table) throws SQLException {
		IDefaultBaseLearnerExperimentConfig cfg = (IDefaultBaseLearnerExperimentConfig)ConfigFactory.create(IDefaultBaseLearnerExperimentConfig.class).loadPropertiesFromFile(new File("conf/experiments/defaultparams/preprocessor.conf"));
		List<String> fieldDescriptors = cfg.getResultFields().stream().filter(n -> n.contains("before")).collect(Collectors.toList());
		List<String> fieldNames = new ArrayList<>();
		Map<String, String> fieldTypes = new HashMap<>();
		fieldTypes.put("eval_id", "int(8)");
		fieldNames.add("openmlid");
		fieldTypes.put("openmlid", "int(6)");
		fieldDescriptors.forEach(desc -> {
			String[] parts = desc.split(":");
			String name = parts[0];
			name = "td_" + name.substring(0, name.length() - "_before".length());
			fieldNames.add(name);
			fieldTypes.put(name, parts[1]);
		});
		fieldNames.addAll(Arrays.asList("td_censored", "traintime"));
		fieldTypes.put("td_censored", "tinyint(1)");
		fieldTypes.put("traintime", "int(8)");
		fieldDescriptors.forEach(desc -> {
			String[] parts = desc.split(":");
			String name = parts[0];
			name = "vd_" + name.substring(0, name.length() - "_before".length());
			fieldNames.add(name);
			fieldTypes.put(name, parts[1]);
		});
		fieldNames.addAll(Arrays.asList("vd_censored", "testtime"));
		fieldTypes.put("vd_censored", "tinyint(1)");
		fieldTypes.put("testtime", "int(8)");

		this.adapter.createTable(table, "eval_id", fieldNames, fieldTypes, Arrays.asList());
	}

	public void convertTable(final String fromTable, final String toTable) throws Exception {
		List<IKVStore> rows;
		String qry = "SELECT *, train_end-train_start as traintime, test_end-test_start as testtime FROM `" + fromTable + "` WHERE experiment_id NOT IN (SELECT `eval_id` FROM `" + toTable + "`) AND test_end IS NOT NULL ORDER BY openmlid LIMIT 1";
		do {
			rows = this.adapter.getResultsOfQuery(qry);
			if (!rows.isEmpty()) {
				Map<String, Object> map = new HashMap<>();
				map.put("eval_id", rows.get(0).get("experiment_id"));
				try {
					this.adapter.insert(toTable, map);
				}
				catch (SQLException e) {
					System.err.println("warning, entry exists: going to next ...");
					continue;
				}
				Map<String, Object> update = this.transformRow(rows.get(0));
				if (update != null) {
					int affectedRows = this.adapter.update(toTable, update, map);
					if (affectedRows != 1) {
						System.err.println("Update failed!");
					}
				}
				else {
					System.err.println("Invalid entry, not updating.");
				}
			}
		}
		while (!rows.isEmpty());
		System.exit(0);

		/* now insert implicitly censored data */
		qry = "SELECT t1.*, t2.eval_id, t2.traintime FROM (SELECT * FROM `" + fromTable + "` WHERE experiment_id NOT IN (SELECT `eval_id` FROM `" + toTable +"`) AND openmlid NOT IN (40927, 41064, 41991, 41026) AND exception LIKE '%Experiment canceled due to lower bound on%' ORDER BY openmlid LIMIT 1000) as t1 NATURAL JOIN (SELECT * FROM `" + toTable + "` WHERE censored = 1) as t2 LIMIT 1 ";
		do {
			rows = this.adapter.getResultsOfQuery(qry);
			if (!rows.isEmpty()) {
				Map<String, Object> map = new HashMap<>();
				map.put("eval_id", rows.get(0).get("experiment_id"));
				try {
					this.adapter.insert(toTable, map);
				}
				catch (SQLException e) {
					System.err.println("warning, entry exists: going to next 	...");
					continue;
				}
				Map<String, Object> update = this.transformCoupledCensoredRow(rows.get(0));
				if (update != null) {
					int affectedRows = this.adapter.update(toTable, update, map);
					if (affectedRows != 1) {
						System.err.println("Update failed!");
					}
				}
				else {
					System.err.println("Invalid entry, not updating.");
				}
			}
		}
		while (!rows.isEmpty());
		System.out.println("No more rows, finishing.");
	}

	private Map<String, Object> transformRow(final IKVStore row) throws Exception {

		/* create map */
		Map<String, Object> map = new HashMap<>();
		map.put("eval_id", row.get("experiment_id"));
		map.put("openmlid", row.get("openmlid"));

		/* if all times are available */
		SimpleDateFormat format = new SimpleDateFormat("HH:mm:ss");
		if (row.getAsString("test_end") != null) {
			map.put("traintime", row.getAsString("traintime"));
			map.put("testtime", row.getAsString("testtime"));
			map.put("td_censored", 0);
			map.put("vd_censored", 0);
		}
		else {

			/* if at least train times are available */
			if (row.getAsString("train_end") != null) {
				Date trainStart = format.parse(row.getAsString("train_start"));
				Date trainEnd = format.parse(row.getAsString("train_end"));
				long trainTime = (trainEnd.getTime() - trainStart.getTime())/ 1000;
				map.put("traintime", trainTime);
				map.put("td_censored", row.getAsString("exception").contains("AlgorithmTimeoutedException") ? 1 : 0);
				System.out.println(map);
			}

			/* otherwise, if at least the train start time is available, censor the data */
			else if (row.getAsString("train_start") != null) {
				System.out.println(row);
			}

			/* otherwise, ignore the data */
			else {
				return null;
			}
		}

		/* create dataset description */
		this.putDatasetDescription(row.getAsInt("openmlid"), row.getAsInt("datapoints"), row.getAsInt("seed"), map);
		return map;
	}

	private Map<String, Object> transformCoupledCensoredRow(final IKVStore row) throws Exception {

		/* create map */
		Map<String, Object> map = new HashMap<>();
		map.put("eval_id", row.get("experiment_id"));
		map.put("openmlid", row.get("openmlid"));
		map.put("traintime", row.getAsInt("traintime"));
		map.put("censored", 1);
		this.putDatasetDescription(row.getAsInt("openmlid"), row.getAsInt("datapoints"), row.getAsInt("seed"), map);
		return map;
	}

	private void putDatasetDescription(final int openmlid, final int datapoints, final long seed, final Map<String, Object> map) throws Exception {
		ILabeledDataset<?> ds = (openmlid == this.lastDatasetId) ? this.lastDataset : OpenMLDatasetReader.deserializeDataset(openmlid);
		this.lastDataset = ds;
		this.lastDatasetId = openmlid;
		double portion = datapoints * 1.0 / ds.size();
		if (portion >= 1) {
			System.err.println("Canceling information, because number of datapoints is too high!");
			return;
		}
		List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, seed, portion);
		this.basicFeatureGen.setPrefix("td_");
		this.varFeatureGen.setPrefix("td_");
		map.putAll(this.basicFeatureGen.getFeatureRepresentation(split.get(0)));
		map.putAll(this.varFeatureGen.getFeatureRepresentation(split.get(0)));
		this.basicFeatureGen.setPrefix("vd_");
		this.varFeatureGen.setPrefix("vd_");
		map.putAll(this.basicFeatureGen.getFeatureRepresentation(split.get(1)));
		map.putAll(this.varFeatureGen.getFeatureRepresentation(split.get(1)));
	}
}
