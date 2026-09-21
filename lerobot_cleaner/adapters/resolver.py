"""Resolve source adapters into a validated canonical feature schema."""
from .schema import CanonicalFeatureSchema


class FeatureResolver:
    def resolve(self, adapter):
        groups = (tuple(adapter.get_state_features()), tuple(adapter.get_action_features()),
                  tuple(adapter.get_camera_features()))
        for features, modality in zip(groups, ("state", "action", "video")):
            names = set()
            for feature in features:
                if feature.name in names or feature.modality != modality:
                    raise ValueError(f"Duplicate or mismatched feature: {feature.name}")
                names.add(feature.name)
                if modality == "video":
                    if not feature.camera_column:
                        raise ValueError(f"Missing camera column: {feature.name}")
                elif not feature.slices:
                    raise ValueError(f"Missing slices: {feature.name}")
                for part in feature.slices:
                    if not part.column or not 0 <= part.start < part.end:
                        raise ValueError(f"Invalid feature slice: {part}")
        quality = getattr(getattr(adapter, "config", None), "quality", None)
        schema = CanonicalFeatureSchema(
            *groups, raw_vectors=getattr(adapter, "raw_vectors", False),
            state_column=getattr(quality, "state_column", "observation.state"),
            action_column=getattr(quality, "action_column", "action"))
        states = {f.name: f for f in schema.states}
        for action in schema.actions:
            if action.subtract_state:
                state = states.get(action.name.replace("action.", "observation.state.", 1))
                if state is None or state.width != action.width:
                    raise ValueError("subtract_state requires matching state dimensions")
        return schema
