package tpami.basealgorithmlearning.datagathering.preprocessing;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;

import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;
import ai.libs.jaicore.experiments.IExperimentDatabaseHandle;
import ai.libs.jaicore.experiments.IExperimentSetConfig;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import tpami.basealgorithmlearning.IConfigContainer;
import tpami.basealgorithmlearning.datagathering.ILearnerExperimentConfig;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;

public class PreprocessorConfigContainer implements IConfigContainer {

	private final ILearnerExperimentConfig config;
	private final IExperimentDatabaseHandle databaseHandle;
	private final IDatabaseAdapter adapter;

	private final String searcher;
	private final String evaluator;

	public PreprocessorConfigContainer(final String dbconfig, final String searcher, final String evaluator) throws Exception {

		/* get experiment configuration */
		AttributeSelection as = new AttributeSelection();
		as.setSearch(ASSearch.forName(searcher, null));
		as.setEvaluator(ASEvaluation.forName(evaluator, null));
		this.config = ConfigFactory.create(ILearnerExperimentConfig.class);
		this.config.loadPropertiesFromFile(new File("conf/experiments/defaultparams/preprocessor.conf"));

		/* setup database connection */
		// IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		IDatabaseConfig dbConfig = ConfigFactory.create(IDatabaseConfig.class);
		dbConfig.loadPropertiesFromFile(new File(dbconfig));
		this.adapter = DatabaseAdapterFactory.get(dbConfig);
		this.databaseHandle = new ExperimenterMySQLHandle(this.adapter, "evaluations_pp_" + searcher.toLowerCase() + "_" + evaluator.toLowerCase());
		this.searcher = searcher;
		this.evaluator = evaluator;
	}

	public ILearnerExperimentConfig getConfig() {
		return this.config;
	}

	@Override
	public IExperimentDatabaseHandle getDatabaseHandle() {
		return this.databaseHandle;
	}

	public String getSearcher() {
		return this.searcher;
	}

	public String getEvaluator() {
		return this.evaluator;
	}

	@Override
	public IDatabaseAdapter getAdapter() {
		return this.adapter;
	}

	@Override
	public IExperimentSetConfig getExperimentSetConfig() {
		return this.getConfig();
	}
}
