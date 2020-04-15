package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.google.common.eventbus.EventBus;

public class EventBusHolder {

	private static Map<String, EventBus> wrapperToEventBusHolder = new HashMap<>();
	private static Set<String> trainedMetaLearners = new HashSet<>();

	public static EventBus getEventBusForWrapper(final LeakingBaselearnerWrapper wrapper) {
		return wrapperToEventBusHolder.get(wrapper.getRandomString());
	}

	public static void registerEventBus(final String randomString, final EventBus eventBus) {
		wrapperToEventBusHolder.put(randomString, eventBus);
	}

	public static void publishFactThatMetaLearnerHasFinishedTraining(final String randomString) {
		trainedMetaLearners.add(randomString);
	}

	public static boolean isMetaLearnerTrained(final String randomString) {
		return trainedMetaLearners.contains(randomString);
	}
}
