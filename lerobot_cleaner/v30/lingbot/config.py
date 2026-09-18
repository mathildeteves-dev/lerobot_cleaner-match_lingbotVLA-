"""Static LingBot mapping validation against real dataset feature dimensions."""

import json
from pathlib import Path

import numpy as np
import yaml


def validate_mapping(dataset: Path, robot_config: Path, train_config: Path) -> dict:
    info = json.loads((Path(dataset) / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("LingBot configuration expects LeRobot v3.0")
    robot = yaml.safe_load(Path(robot_config).read_text(encoding="utf-8"))
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
    dimensions = {}
    for section, prefix in [("states", "observation.state."), ("actions", "action.")]:
        if not isinstance(robot[section], list) or not robot[section]:
            raise ValueError(f"Empty or invalid {section}")
        for item in robot[section]:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError("Use explicit origin_keys mappings")
            target, mapping = next(iter(item.items()))
            joint = target.removeprefix(prefix)
            if not target.startswith(prefix) or joint not in joints or target in dimensions:
                raise ValueError(f"Invalid/duplicate joint target: {target}")
            width = 0
            origins = mapping["origin_keys"]
            if not isinstance(origins, list) or not origins:
                raise ValueError(f"origin_keys must be a nonempty list: {target}")
            for origin in origins:
                if not isinstance(origin, dict) or len(origin) != 1:
                    raise ValueError("Each origin must contain one feature/slice")
                key, bounds = next(iter(origin.items()))
                if key not in info["features"]:
                    raise ValueError(f"Missing source feature: {key}")
                feature = info["features"][key]
                if not feature["dtype"].startswith("float") or len(feature["shape"]) != 1:
                    raise ValueError(f"Expected floating-point vector: {key}")
                start, end = bounds["start"], bounds["end"]
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= int(np.prod(feature["shape"]))
                ):
                    raise ValueError(f"Slice out of bounds: {key}[{start}:{end}]")
                width += end - start
            if width > joints[joint]:
                raise ValueError(f"Mapped width exceeds joint capacity: {target}")
            dimensions[target] = width
            if section == "actions":
                if not isinstance(mapping.get("subtract_state"), bool):
                    raise ValueError(f"Explicit subtract_state boolean required: {target}")
                if mapping["subtract_state"] and (
                    "effector.position" in target or "end.position" in target
                ):
                    raise ValueError(f"LingBot forbids subtract_state for {target}")
                if (
                    mapping["subtract_state"]
                    and dimensions.get("observation.state." + joint) != width
                ):
                    raise ValueError("subtract_state requires a matching state dimension")
    image_targets = set()
    for item in robot["images"]:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError("Expected explicit image mapping")
        target, mapping = next(iter(item.items()))
        key = mapping["origin_keys"]
        if key not in info["features"] or info["features"][key]["dtype"] != "video":
            raise ValueError(f"Missing video feature: {key}")
        name = target.removeprefix("observation.images.")
        if (
            not target.startswith("observation.images.")
            or name not in cameras
            or name in image_targets
        ):
            raise ValueError(f"Invalid/duplicate camera target: {target}")
        image_targets.add(name)
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
