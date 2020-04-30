package tpami.safeguard.api;

public enum EMetaFeature {

	NUM_INSTANCES("numinstances"), NUM_ATTRIBUTES("numattributes");

	private String fieldName;

	private EMetaFeature(final String fieldName) {
		this.fieldName = fieldName;
	}

	@Override
	public String toString() {
		return this.fieldName;
	}

}
