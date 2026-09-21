"""Static LingBot mapping validation against real dataset feature dimensions."""

from pathlib import Path

import yaml


def validate_mapping(dataset: Path, robot_config: Path, train_config: Path) -> dict:
    train = yaml.safe_load(Path(train_config).read_text(encoding="utf-8"))
    data = train["data"]
    if data["data_name"] != Path(robot_config).stem:
        raise ValueError("data_name must match robot config filename")
    joints = {}
    for item in data["joints"]:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Each joint entry must contain one name/dimension")
        name, size = next(iter(item.items()))
        if name in joints or not isinstance(size, int) or size <= 0:
            raise ValueError("Invalid/duplicate joint capacity")
        joints[name] = size
    cameras = data["cameras"]
    if len(cameras) != len(set(cameras)):
        raise ValueError("Duplicate camera targets")
    from lerobot_cleaner.adapters.lingbot import LingBotAdapter
    with LingBotAdapter(dataset, robot_config) as adapter:
        dimensions = {}
        for feature in adapter.get_state_features() + adapter.get_action_features():
            prefix = "observation.state." if feature.modality == "state" else "action."
            joint = feature.name.removeprefix(prefix)
            if joint not in joints:
                raise ValueError(f"Invalid/duplicate joint target: {feature.name}")
            if feature.width > joints[joint]:
                raise ValueError(f"Mapped width exceeds joint capacity: {feature.name}")
            dimensions[feature.name] = feature.width
        image_targets = {feature.name.removeprefix("observation.images.") for feature in adapter.get_camera_features()}
        if image_targets != set(cameras):
            raise ValueError("Every configured camera must be mapped")
        for section in ["state", "action"]:
            if sum(joints.values()) > train["train"][f"max_{section}_dim"]:
                raise ValueError(f"Joint capacities exceed max_{section}_dim")
        return {
            "data_name": data["data_name"],
            "mapped_dimensions": dimensions,
            "cameras": cameras,
            "validation": "static_schema_only",
            "warning": "Does not verify controller semantics, checkpoints or LingBot runtime loading.",
        }
