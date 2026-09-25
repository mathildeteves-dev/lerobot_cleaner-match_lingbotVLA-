"""Explicit generic semantics over official LeRobot storage, without model config."""
from pathlib import Path
from .lerobot_v3 import LeRobotAdapter
from .mapping import load_mapping
from .resolver import FeatureResolver


class GenericMappingAdapter(LeRobotAdapter):
    def __init__(self, root, mapping_config, config=None, *, info=None, writer=None, storage=None):
        super().__init__(root, config, info=info, writer=writer, storage=storage)
        self.mapping_config = Path(mapping_config).resolve()
        try:
            self.mapping_specs = load_mapping(self.mapping_config)
            self._mapped = FeatureResolver().resolve_mapping(self.mapping_specs, self.info["features"])
            self.get_feature_schema()
        except Exception:
            self.close()
            raise

    def get_state_features(self):
        return tuple(feature for feature in self._mapped if feature.modality == "state")

    def get_action_features(self):
        return tuple(feature for feature in self._mapped if feature.modality == "action")

    def describe(self):
        return {**super().describe(), "mapping_config": str(self.mapping_config),
                "feature_order": "canonical target order and ordered source concatenation",
                "mapping_grammar": "generic canonical state/action"}
