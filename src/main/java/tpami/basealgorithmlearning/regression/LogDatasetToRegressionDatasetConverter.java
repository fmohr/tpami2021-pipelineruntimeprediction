package tpami.basealgorithmlearning.regression;

import java.io.File;
import java.sql.SQLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.sql.rest.IRestDatabaseConfig;
import ai.libs.jaicore.db.sql.rest.RestSqlAdapter;
import ai.libs.jaicore.ml.core.dataset.serialization.OpenMLDatasetReader;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;

/**
 * Turns a dataset with observations of a classifier into a regression dataset with the runtime information
 *
 * @author Felix Mohr
 */
class LogDatasetToRegressionDatasetConverter {

	private final IDatabaseAdapter adapter;
	private DatasetFeatureGenerator trainGen = new DatasetFeatureGenerator("td");
	private DatasetFeatureGenerator validationGen = new DatasetFeatureGenerator("vd");
	private ILabeledDataset<?> lastDataset;
	private int lastDatasetId;

	public static void main(final String[] args) throws Exception {
		LogDatasetToRegressionDatasetConverter con = new LogDatasetToRegressionDatasetConverter(new RestSqlAdapter((IRestDatabaseConfig)ConfigFactory.create(IRestDatabaseConfig.class).loadPropertiesFromFile(new File(args[0]))));
		String classifier = args[1];
		con.convertTable("evaluations_classifiers_" + classifier, "regression_classifiers_" + classifier);
	}

	public LogDatasetToRegressionDatasetConverter(final IDatabaseAdapter adapter) {
		this.adapter = adapter;
	}

	public void convertTable(final String fromTable, final String toTable) throws SQLException, ParseException, DatasetDeserializationFailedException, SplitFailedException, InterruptedException {
		List<IKVStore> rows;
		String qry = "SELECT * FROM `" + fromTable + "` WHERE experiment_id NOT IN (SELECT `eval_id` FROM `" + toTable + "`) ORDER BY openmlid LIMIT 1";
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
	}

	private Map<String, Object> transformRow(final IKVStore row) throws ParseException, DatasetDeserializationFailedException, SplitFailedException, InterruptedException {

		/* create map */
		Map<String, Object> map = new HashMap<>();
		map.put("eval_id", row.get("experiment_id"));
		map.put("openmlid", row.get("openmlid"));

		/* if all times are available */
		SimpleDateFormat format = new SimpleDateFormat("HH:mm:ss");
		if (row.getAsString("test_end") != null) {
			Date trainStart = format.parse(row.getAsString("train_start"));
			Date trainEnd = format.parse(row.getAsString("train_end"));
			Date testStart = format.parse(row.getAsString("test_start"));
			Date testEnd = format.parse(row.getAsString("test_end"));
			long trainTime = (trainEnd.getTime() - trainStart.getTime())/ 1000;
			long testTime = (testEnd.getTime() - testStart.getTime()) / 1000;
			map.put("traintime", trainTime);
			map.put("testtime", testTime);
			map.put("censored", 0);
		}
		else {

			/* if at least train times are available */
			if (row.getAsString("train_end") != null) {
				Date trainStart = format.parse(row.getAsString("train_start"));
				Date trainEnd = format.parse(row.getAsString("train_end"));
				long trainTime = (trainEnd.getTime() - trainStart.getTime())/ 1000;
				map.put("traintime", trainTime);
				map.put("censored", row.getAsString("exception").contains("AlgorithmTimeoutedException") ? 1 : 0);
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
		int id = row.getAsInt("openmlid");
		ILabeledDataset<?> ds = (id == this.lastDatasetId) ? this.lastDataset : OpenMLDatasetReader.deserializeDataset(id);
		this.lastDataset = ds;
		this.lastDatasetId = id;
		double portion = row.getAsInt("datapoints") * 1.0 / ds.size();
		if (portion >= 1) {
			System.err.println("Canceling information, because number of datapoints is too high!");
			return null;
		}
		List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(ds, row.getAsInt("seed"), portion);
		map.putAll(this.trainGen.getFeatureRepresentation(split.get(0)));
		map.put("vd_instances", split.get(1).size());
		return map;
	}
}
