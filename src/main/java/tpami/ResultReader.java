package tpami;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;

import ai.libs.jaicore.basic.sets.SetUtil;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.SQLAdapter;
import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.evaluation.evaluator.PredictionDiff;

public class ResultReader {
	public static void main(final String[] args) throws Exception {

		/* establish DB connection */
		SQLAdapter adapter = new SQLAdapter((IDatabaseConfig)ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("dbcon.conf")));
		final String table = "mlplanmlj2019reeval_aggregate";

		/* parse SQL file */
		Pattern p = Pattern.compile("'([^']*)'");
		try (BufferedReader br = new BufferedReader(new FileReader(new File("mlplanmlj2019reeval.sql")))) {
			String line;
			while ((line = br.readLine()) != null) {
				String[] parts = line.split(",");
				if (!line.startsWith("(")) {
					continue;
				}
				Matcher m = p.matcher(line);
				int counter = 0;
				int openmlid = Integer.parseInt(parts[2].trim());
				List<?> predictions = null;
				List<?> groundTruth = null;
				String data = null;
				String pipeline = null;
				String exception = null;
				Date trainStart = null;
				Date trainEnd = null;
				Date testStart = null;
				Date testEnd = null;
				int seed = Integer.parseInt(parts[3].trim());

				SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

				while (m.find()) {
					String match = m.group(1);
					switch (counter) {
					case 1:
						data = match.replace("\\\"", "\"").replace("\\\\", "\\");
						break;
					case 4:
						pipeline = match.replace("\\\"", "\"").replace("\\\\", "\\");
						break;
					case 5:
						trainStart = format.parse(match);
						break;
					case 6:
						trainEnd = format.parse(match);
						break;
					case 7:
						if (match.toLowerCase().contains("exception")) {
							exception = match;
							break;
						}
						testStart = format.parse(match);
						break;
					case 8:
						testEnd = format.parse(match);
						break;
					case 9:
						if (match.toLowerCase().contains("exception")) {
							exception = match;
							break;
						}
						groundTruth = SetUtil.unserializeList(match);
						break;
					case 10:
						predictions = SetUtil.unserializeList(match);
						break;
					case 11:
					case 12:
						System.out.println(counter + ": " + match);
					}
					counter ++;
				}
				if (groundTruth != null) {
					PredictionDiff<? extends Integer, ? extends ISingleLabelClassification> diff = (PredictionDiff<? extends Integer, ? extends ISingleLabelClassification>)new PredictionDiff<>(groundTruth, predictions);
					double errorRate = EClassificationPerformanceMeasure.ERRORRATE.loss(diff);
					long trainTime = (trainEnd.getTime() - trainStart.getTime())/ 1000;
					long testTime = (testEnd.getTime() - testStart.getTime()) / 1000;
					Map<String, Object> map = new HashMap<>();
					map.put("openmlid", openmlid);
					map.put("seed", seed);
					map.put("evaluationinputdata", data);
					map.put("pipeline", pipeline);
					map.put("traintime", trainTime);
					map.put("testtime", testTime);
					map.put("errorrate", errorRate);
					try {
						adapter.insert(table, map);
					}
					catch (Exception e) {
						System.err.println(map);
						throw e;
					}
				}
				else if (exception != null) {
					long trainTime = (trainEnd.getTime() - trainStart.getTime())/ 1000;
					Map<String, Object> map = new HashMap<>();
					map.put("openmlid", openmlid);
					map.put("seed", seed);
					map.put("evaluationinputdata", data);
					map.put("pipeline", pipeline);
					map.put("traintime", trainTime);
					if (testEnd != null) {
						long testTime = (testEnd.getTime() - testStart.getTime()) / 1000;
						map.put("testtime", testTime);
					}
					map.put("exception", exception.replace("\\n", "\n"));
					try {
						adapter.insert(table, map);
					}
					catch (Exception e) {
						System.err.println(map);
						throw e;
					}
				}
			}
		}
	}
}
