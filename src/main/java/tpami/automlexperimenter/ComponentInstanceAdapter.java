package tpami.automlexperimenter;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentInstance;

public class ComponentInstanceAdapter {

	private static final String L_COMPONENT = "component";
	private static final String L_NAME = "name";
	private static final String L_PARAM_VALUES = "parameterValues";
	private static final String L_SAT_REQ_IFACE = "satisfactionOfRequiredInterfaces";

	private Collection<IComponent> components;

	public ComponentInstanceAdapter(final Collection<IComponent> components) {
		this.components = components;
	}

	public ComponentInstanceAdapter() {
		this(new LinkedList<>());
	}

	public String componentInstanceToString(final IComponentInstance ci) throws JsonProcessingException {
		if (ci == null) {
			return "null";
		}
		ObjectMapper mapper = new ObjectMapper();
		return mapper.writeValueAsString(this.componentInstanceToMap(ci));
	}

	public Map<String, Object> componentInstanceToMap(final IComponentInstance ci) {
		Map<String, Object> ciMap = new HashMap<>();
		ciMap.put(L_COMPONENT, this.componentToString(ci.getComponent()));
		ciMap.put(L_PARAM_VALUES, ci.getParameterValues());

		Map<String, Object> satisfactionOfRequiredInterfaces = new HashMap<>();
		ci.getSatisfactionOfRequiredInterfaces().entrySet().stream().forEach(x -> satisfactionOfRequiredInterfaces.put(x.getKey(), this.componentInstanceToMap(x.getValue().get(0))));
		ciMap.put(L_SAT_REQ_IFACE, satisfactionOfRequiredInterfaces);

		return ciMap;
	}

	private Map<String, Object> componentToString(final IComponent comp) {
		Map<String, Object> componentMap = new HashMap<>();
		componentMap.put(L_NAME, comp.getName());
		return componentMap;
	}

	public ComponentInstance stringToComponentInstance(final String ciString) throws IOException {
		JsonNode root = new ObjectMapper().readTree(ciString);
		return this.readComponentInstanceFromJson(root);
	}

	private ComponentInstance readComponentInstanceFromJson(final JsonNode node) throws JsonProcessingException {
		String componentName = node.get(L_COMPONENT).get(L_NAME).asText();
		IComponent component = this.components.stream().filter(x -> x.getName().equals(componentName)).findAny().get();

		Map<String, String> parameterValues = new HashMap<>();
		Iterator<String> parameterValueIt = node.get(L_PARAM_VALUES).fieldNames();
		while (parameterValueIt.hasNext()) {
			String fieldName = parameterValueIt.next();
			parameterValues.put(fieldName, node.get(L_PARAM_VALUES).get(fieldName).asText());
		}

		Map<String, List<IComponentInstance>> satisfactionOfRequiredInterfaces = new HashMap<>();
		if (node.get(L_SAT_REQ_IFACE) != null) {
			Iterator<String> satReqIfaceIt = node.get(L_SAT_REQ_IFACE).fieldNames();
			while (satReqIfaceIt.hasNext()) {
				String ifaceName = satReqIfaceIt.next();
				satisfactionOfRequiredInterfaces.put(ifaceName, Arrays.asList(this.readComponentInstanceFromJson(node.get(L_SAT_REQ_IFACE).get(ifaceName))));
			}
		}

		return new ComponentInstance(component, parameterValues, satisfactionOfRequiredInterfaces);
	}

}
