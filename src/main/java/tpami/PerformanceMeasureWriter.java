package tpami;

import java.io.File;
import java.sql.SQLException;
import java.util.List;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;

public class PerformanceMeasureWriter {
	public static void main(final String[] args) throws SQLException {
		IDatabaseConfig dbConfigExt = (IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("dbcon-ext.conf"));
		IDatabaseConfig dbConfigLocal = (IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("dbcon.conf"));
		System.out.println(dbConfigExt + "\n\n" + dbConfigLocal);
		System.out.println("Reading data from table " + dbConfigExt.getDBTableName() + " in DB " + dbConfigExt.getDBDatabaseName() + " on server " + dbConfigExt.getDBHost());
		try (SQLAdapter dbAdapter = new SQLAdapter(dbConfigExt)){
			List<IKVStore> stores = dbAdapter.getRowsOfTable(dbConfigExt.getDBTableName());
			System.out.println("done. Read " + stores.size() + " data rows.");
			try (IDatabaseAdapter dbAdapterLocal = new SQLAdapter(dbConfigLocal)){
				for (IKVStore store : stores) {
					dbAdapter.insert(dbConfigLocal.getDBTableName(), store);
				}
			}
		}
	}
}
