"""LingBot robot YAML semantics over native v3 storage; no modality.json access."""
from pathlib import Path

import yaml

from .schema import FeatureSchema, FeatureSlice
from .lerobot_v3 import LeRobotV3Adapter


class LingBotAdapter(LeRobotV3Adapter):
    def __init__(self, root, robot_config, config=None, *, info=None, writer=None, storage=None):
        super().__init__(root, config, info=info, writer=writer, storage=storage)
        self.robot_config = Path(robot_config).resolve()
        robot = yaml.safe_load(self.robot_config.read_text(encoding="utf-8"))
        if not isinstance(robot, dict) or set(robot) != {"states", "actions", "images"}:
            raise ValueError("Robot config must define states, actions and images")
        self._features = {}
        for section, modality, prefix in [("states", "state", "observation.state."),
                                           ("actions", "action", "action."),
                                           ("images", "video", "observation.images.")]:
            features = []
            seen = set()
            if not isinstance(robot[section], list) or not robot[section]:
                raise ValueError(f"Robot config requires nonempty {section}")
            for entry in robot[section]:
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ValueError(f"Invalid feature entry in {section}")
                name, mapping = next(iter(entry.items()))
                if name in seen or not name.startswith(prefix):
                    raise ValueError(f"Duplicate/invalid target feature: {name}")
                seen.add(name)
                if modality == "video":
                    column = mapping["origin_keys"]
                    if not isinstance(column, str) or self.info["features"].get(column, {}).get("dtype") != "video":
                        raise ValueError(f"Missing video source: {column}")
                    features.append(FeatureSchema(name, modality, camera_column=column))
                    continue
                slices = []
                origins = mapping["origin_keys"]
                if not isinstance(origins, list) or not origins:
                    raise ValueError(f"Empty origin_keys for {name}")
                for origin in origins:
                    if not isinstance(origin, dict) or len(origin) != 1:
                        raise ValueError(f"Invalid source slice: {name}")
                    column, bounds = next(iter(origin.items()))
                    spec = self.info["features"].get(column)
                    if spec is None or not spec["dtype"].startswith("float") or len(spec["shape"]) != 1:
                        raise ValueError(f"Expected floating-point vector: {column}")
                    start, end = bounds["start"], bounds["end"]
                    if type(start) is not int or type(end) is not int or not 0 <= start < end <= spec["shape"][0]:
                        raise ValueError(f"Slice out of bounds: {column}[{start}:{end}]")
                    slices.append(FeatureSlice(column, start, end))
                subtract = mapping.get("subtract_state", False)
                if modality == "action" and type(mapping.get("subtract_state")) is not bool:
                    raise ValueError(f"Explicit subtract_state boolean required: {name}")
                features.append(FeatureSchema(name, modality, tuple(slices), subtract_state=subtract))
            self._features[modality] = tuple(features)
        states = {feature.name: feature for feature in self._features["state"]}
        for feature in self._features["action"]:
            if feature.subtract_state:
                state = states.get(feature.name.replace("action.", "observation.state.", 1))
                if state is None or state.width != feature.width:
                    raise ValueError("subtract_state requires matching state dimensions")
                if "effector.position" in feature.name or "end.position" in feature.name:
                    raise ValueError(f"LingBot forbids subtract_state for {feature.name}")

    def get_state_features(self):
        return self._features["state"]

    def get_action_features(self):
        return self._features["action"]

    def get_camera_features(self):
        return self._features["video"]
