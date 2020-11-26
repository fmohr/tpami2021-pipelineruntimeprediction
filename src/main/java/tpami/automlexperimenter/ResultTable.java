package tpami.automlexperimenter;

import java.io.File;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection.EGroupMethod;
import ai.libs.jaicore.basic.kvstore.KVStoreSequentialComparator;
import ai.libs.jaicore.basic.kvstore.KVStoreStatisticsUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreUtil;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;

public class ResultTable {

	private static IDatabaseConfig DBC;

	public static void main(final String[] args) {
		DBC = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("automlexperimenter.properties"));
		try (IDatabaseAdapter adapter = DatabaseAdapterFactory.get(DBC)) {
			KVStoreCollection safeguard1h = KVStoreUtil.readFromMySQLQuery(adapter, "SELECT * FROM cont_jobs_mlplan_safeguard_1h WHERE loss IS NOT NULL", new HashMap<>());
			KVStoreCollection vanilla1h = KVStoreUtil.readFromMySQLQuery(adapter, "SELECT * FROM cont_jobs_mlplan_1h WHERE loss IS NOT NULL", new HashMap<>());

			KVStoreCollection safeguard24h = KVStoreUtil.readFromMySQLQuery(adapter, "SELECT * FROM cont_jobs_mlplan_safeguard_24h WHERE loss IS NOT NULL", new HashMap<>());
			KVStoreCollection vanilla24h = KVStoreUtil.readFromMySQLQuery(adapter, "SELECT * FROM cont_jobs_mlplan_24h WHERE loss IS NOT NULL", new HashMap<>());

			KVStoreCollection merged1h = new KVStoreCollection();
			merged1h.addAll(safeguard1h);
			merged1h.addAll(vanilla1h);

			merged1h.stream().forEach(x -> x.put("approach", x.getAsString("algorithmmode") + "-" + x.getAsString("timeout")));

			KVStoreCollection merged24h = new KVStoreCollection();
			merged1h.addAll(safeguard24h);
			merged1h.addAll(vanilla24h);

			Map<String, EGroupMethod> groupMethod = new HashMap<>();
			groupMethod.put("loss", EGroupMethod.AVG);
			KVStoreCollection grouped1h = merged1h.group(new String[] { "approach", "dataset", "timeout" }, groupMethod);
			KVStoreCollection grouped24h = merged24h.group(new String[] { "approach", "dataset", "timeout" }, groupMethod);

			KVStoreStatisticsUtil.bestWilcoxonSignedRankTest(grouped1h, "dataset", "approach", "seed", "loss_list", "sig");
			KVStoreStatisticsUtil.bestWilcoxonSignedRankTest(grouped24h, "dataset", "approach", "seed", "loss_list", "sig");

			KVStoreCollection groupedAll = new KVStoreCollection();
			groupedAll.addAll(grouped1h);
			groupedAll.addAll(grouped24h);

			groupedAll.stream().forEach(x -> x.put("loss", ValueUtil.valueToString(x.getAsDouble("loss") * 100, 2)));
			groupedAll.stream().forEach(x -> x.put("loss_stdDev", ValueUtil.valueToString(x.getAsDouble("loss_stdDev") * 100, 2)));
			groupedAll.stream().forEach(x -> x.put("entry", x.getAsString("loss") + " $\\pm$ " + x.getAsString("loss_stdDev")));

			groupedAll.sort(new KVStoreSequentialComparator("timeout", "algorithmmode", "dataset"));

			String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(groupedAll, "dataset", "approach", "entry");
			System.out.println(latexTable);

		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

}
