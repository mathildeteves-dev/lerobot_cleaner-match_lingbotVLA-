"""LingBot robot grammar normalized before canonical feature resolution."""
from pathlib import Path
from .resolver import FeatureResolver
from .lingbot_config import LingBotRobotConfigSchema
from .lerobot_v3 import LeRobotV3Adapter


class LingBotAdapter(LeRobotV3Adapter):
    def __init__(self, root, robot_config, config=None, *, info=None, writer=None, storage=None):
        super().__init__(root, config, info=info, writer=writer, storage=storage)
        self.robot_config = Path(robot_config).resolve()
        try:
            self.robot_schema = LingBotRobotConfigSchema.from_yaml(self.robot_config, self.info["features"])
            self.mapping_specs = tuple(f.normalized() for f in self.robot_schema.states + self.robot_schema.actions)
            self._features = {
                "state": tuple(f.canonical() for f in self.robot_schema.states),
                "action": tuple(f.canonical() for f in self.robot_schema.actions),
                "visual": tuple(FeatureResolver().resolve_visual(f.canonical(), self.info["features"]) for f in self.robot_schema.images),
            }
            self.get_feature_schema()
        except Exception:
            self.close()
            raise

    def get_state_features(self):
        return self._features["state"]

    def get_action_features(self):
        return self._features["action"]

    def get_camera_features(self):
        return self._features["visual"]

    def describe(self):
        return {**super().describe(), "lingbot_grammar_revision": self.robot_schema.upstream_revision,
                "convert_from_state": "upstream marker only; explicit origin_keys defines the source",
                "feature_order": "robot YAML mapping order; training padding order comes from data_config.joints"}
