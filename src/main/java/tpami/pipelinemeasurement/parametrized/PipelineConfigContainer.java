package tpami.pipelinemeasurement.parametrized;

import tpami.basealgorithmlearning.AConfigContainer;

public class PipelineConfigContainer extends AConfigContainer {

	public PipelineConfigContainer(final String databaseConfigFile) throws ClassNotFoundException {
		super("conf/experiments/parametrized/pipelines.conf", databaseConfigFile, "pipelines");
	}
}
