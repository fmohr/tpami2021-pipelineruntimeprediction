package tpami.basealgorithmlearning.datagathering.preprocessing;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import tpami.basealgorithmlearning.datagathering.classification.defaultparams.IDefaultBaseLearnerExperimentConfig;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;

public class PreprocessorConfigContainer {

	private final IDefaultBaseLearnerExperimentConfig config;
	private final IExperimentDatabaseHandle databaseHandle;

	public PreprocessorConfigContainer(final String dbconfig, final String searcher, final String evaluator) throws Exception {

		/* get experiment configuration */
		AttributeSelection as = new AttributeSelection();
		as.setSearch(ASSearch.forName(searcher, null));
		as.setEvaluator(ASEvaluation.forName(evaluator, null));
		this.config = ConfigFactory.create(IDefaultBaseLearnerExperimentConfig.class);
		this.config.loadPropertiesFromFile(new File("conf/experiments/defaultparams/preprocessor.conf"));

		/* setup database connection */
		//		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		dbConfig.loadPropertiesFromFile(new File(dbconfig));
		final IDatabaseAdapter adapter = DatabaseAdapterFactory.get(dbConfig);
		this.databaseHandle = new ExperimenterMySQLHandle(adapter, "evaluations_preprocessors_" + searcher + "_" + evaluator);
	}

	public IDefaultBaseLearnerExperimentConfig getConfig() {
		return this.config;
	}

	public IExperimentDatabaseHandle getDatabaseHandle() {
		return this.databaseHandle;
	}
}
