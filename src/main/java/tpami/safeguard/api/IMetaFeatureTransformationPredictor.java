package tpami.safeguard.api;

import ai.libs.hasco.model.ComponentInstance;
import tpami.safeguard.impl.MetaFeatureContainer;

public interface IMetaFeatureTransformationPredictor {

	public String getComponentName();

	public MetaFeatureContainer predictTransformedMetaFeatures(final ComponentInstance ci, final MetaFeatureContainer metaFeaturesBefore) throws Exception;

}