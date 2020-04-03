package tpami.basealgorithmlearning.datagathering;

import java.text.SimpleDateFormat;
import java.util.Date;

public class DataGatheringUtil {

	public static SimpleDateFormat getDateTimeFormat() {
		return new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");
	}

	public static String formatDate(Date date) {
		return getDateTimeFormat().format(date);
	}
}
