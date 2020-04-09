package tpami.basealgorithmlearning.datagathering.metaalgorithm.defaultparams;

import java.util.HashMap;
import java.util.Map;

import com.google.common.eventbus.EventBus;

public class EventBusHolder {

	public static Map<String, EventBus> wrapperToEventBusHolder = new HashMap<>();

	public static EventBus getEventBusForWrapper(LeakingBaselearnerWrapper wrapper) {
		return wrapperToEventBusHolder.get(wrapper.getRandomString());
	}

	public static void registerEventBus(String randomString, EventBus eventBus) {
		wrapperToEventBusHolder.put(randomString, eventBus);
	}
}
