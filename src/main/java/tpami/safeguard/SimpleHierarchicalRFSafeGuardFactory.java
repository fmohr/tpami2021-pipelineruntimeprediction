package tpami.safeguard;

import java.util.List;

import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.ISupervisedLearnerEvaluator;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;

import ai.libs.jaicore.ml.core.evaluation.evaluator.MonteCarloCrossValidationEvaluator;
import ai.libs.mlplan.safeguard.IEvaluationSafeGuardFactory;

public class SimpleHierarchicalRFSafeGuardFactory implements IEvaluationSafeGuardFactory {

	private int[] excludeOpenMLDatasets;
	private ISupervisedLearnerEvaluator<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>> evaluator;
	private ILabeledDataset<?> train;
	private ILabeledDataset<?> test;

	private SimpleHierarchicalRFSafeGuard builtSafeGuard = null;

	public SimpleHierarchicalRFSafeGuardFactory() {
		// intentionally left blank
	}

	@Override
	public SimpleHierarchicalRFSafeGuardFactory withEvaluator(final ISupervisedLearnerEvaluator<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>> evaluator) {
		this.evaluator = evaluator;
		if (this.evaluator instanceof MonteCarloCrossValidationEvaluator) {
			MonteCarloCrossValidationEvaluator mccv = (MonteCarloCrossValidationEvaluator) this.evaluator;
			try {
				IDatasetSplitSet<ILabeledDataset<?>> splitSet = mccv.getSplitGenerator().nextSplitSet();
				List<ILabeledDataset<?>> split = splitSet.getFolds(0);
				this.train = split.get(0);
				this.test = split.get(1);
			} catch (InterruptedException | SplitFailedException e) {
				e.printStackTrace();
			}
		}
		return this;
	}

	public SimpleHierarchicalRFSafeGuardFactory withTrainingDataSet(final ILabeledDataset<?> train) {
		this.train = train;
		return this;
	}

	public SimpleHierarchicalRFSafeGuardFactory withTestDataSet(final ILabeledDataset<?> test) {
		this.test = test;
		return this;
	}

	public SimpleHierarchicalRFSafeGuardFactory withExcludeOpenMLDatasets(final int[] excludeOpenMLDatasets) {
		this.excludeOpenMLDatasets = excludeOpenMLDatasets;
		return this;
	}

	@Override
	public SimpleHierarchicalRFSafeGuard build() throws Exception {
		if (this.builtSafeGuard == null) {
			this.builtSafeGuard = new SimpleHierarchicalRFSafeGuard(this.excludeOpenMLDatasets, this.evaluator, this.train, this.test);
		}
		return this.builtSafeGuard;
	}

	public void setNumCPUs(final int numCPUs) {
		SimpleHierarchicalRFSafeGuard.setNumCPUs(numCPUs);
	}

}
