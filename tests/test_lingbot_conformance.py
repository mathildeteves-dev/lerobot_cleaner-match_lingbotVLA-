"""Behavioral contract against the real, unmodified official FeatureTransform.

Set LINGBOT_ROOT to an official checkout. No mocked upstream algorithms.
Only package __init__ side effects (training/model imports) are bypassed.
"""
import copy
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from lerobot_cleaner.adapters.lingbot import LingBotAdapter
from lerobot_cleaner.adapters.lingbot_config import LingBotRobotConfigSchema

PROJECT = Path(__file__).resolve().parents[1]
FEATURES = {key: {"dtype": "float32", "shape": [size]} for key, size in {
    "observation.state": 6, "observation.state.arm": 6, "action": 6,
    "sensor_b": 4, "action.arm": 6,
}.items()}
FEATURES["observation.images.top"] = {"dtype": "video", "shape": [4, 5, 3]}


def sample():
    return {key: (np.arange(np.prod(spec["shape"]), dtype=np.float32).reshape(spec["shape"]).transpose(2, 0, 1)
                  if spec["dtype"] == "video" else
                  np.arange(3 * spec["shape"][0], dtype=np.float32).reshape(3, -1) + 100 * i)
            for i, (key, spec) in enumerate(FEATURES.items())}


def adapter(tmp_path, robot):
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")
    storage = SimpleNamespace(root=tmp_path, info={"features": FEATURES, "fps": 10})
    return LingBotAdapter(tmp_path, path, storage=storage), path


@pytest.fixture(scope="module")
def upstream():
    root = Path(os.environ.get("LINGBOT_ROOT", PROJECT.parent.parent / "lingbot-vla"))
    source = root / "lingbotvla/data/vla_data/utils.py"
    if not source.is_file():
        pytest.skip("Official source unavailable; set LINGBOT_ROOT. This is not a conformance pass.")
    pytest.importorskip("torch", reason="Official FeatureTransform requires torch")
    pytest.importorskip("einops", reason="Official transform module requires einops")
    package_name = "_cleaner_lingbot_conformance"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source.parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".utils", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module.FeatureTransform
    for key in list(sys.modules):
        if key == package_name or key.startswith(package_name + "."):
            del sys.modules[key]


def official(upstream, path, robot):
    names, cameras = [], []
    for section in ("states", "actions", "images"):
        for entry in robot.get(section, []):
            target = entry if isinstance(entry, str) else next(iter(entry))
            if section == "images":
                cameras.append(target.removeprefix("observation.images."))
            else:
                joint = target.split("observation.state." if section == "states" else "action.")[-1]
                if joint not in names:
                    names.append(joint)
    data = SimpleNamespace(joints=[str({name: 32}) for name in names], cameras=cameras)
    return upstream(path, data, None, None, do_nomalize=False, load_image=True)


CASES = {
    "A_direct_state": {"states": ["observation.state"]},
    "B_direct_image": {"images": ["observation.images.top"]},
    "C_rename_image": {"images": [{"observation.images.renamed": {"origin_keys": "observation.images.top"}}]},
    "D_single_state_string_origin": {"states": [{"state": {"origin_keys": "observation.state"}}]},
    "E_multiple_sources": {"states": [{"state": {"origin_keys": [
        {"sensor_b": {"start": 1, "end": 4}}, {"observation.state": {"start": 0, "end": 2}}]}}]},
    "F_subtract_state": {"states": [{"observation.state.arm": {"origin_keys": "observation.state"}}],
        "actions": [{"action.arm": {"origin_keys": "action", "subtract_state": True}}]},
    "G_convert_from_state_explicit_state": {"states": ["observation.state.arm"],
        "actions": [{"action.arm": {"origin_keys": "observation.state.arm", "convert_from_state": True}}]},
    "G_convert_sliced_state_without_state_target": {"actions": [{"action.arm": {
        "origin_keys": [{"observation.state": {"start": 3, "end": 6}},
                        {"observation.state": {"start": 0, "end": 2}}], "convert_from_state": True}}]},
    "G_convert_flag_does_not_rewire_action": {"states": ["observation.state.arm"],
        "actions": [{"action.arm": {"origin_keys": "action", "convert_from_state": True, "subtract_state": True}}]},
    "H_repeated_slices": {"states": [{"state": {"origin_keys": [
        {"observation.state": {"start": 4, "end": 6}},
        {"observation.state": {"start": 0, "end": 3}},
        {"sensor_b": {"start": 2, "end": 4}}]}}]},
    "inner_mapping_order": {"states": [{"state": {"origin_keys": [
        {"sensor_b": {"start": 1, "end": 3}, "observation.state": {"start": 0, "end": 1}}]}}]},
    "python_negative_clipped_slices": {"states": [{"state": {"origin_keys": [
        {"observation.state": {"start": -3, "end": 99}}]}}]},
    "image_slice_concat": {"images": [{"observation.images.joined": {"origin_keys": [
        {"observation.images.top": {"start": 3, "end": 5}},
        {"observation.images.top": {"start": 0, "end": 2}}]}}]},
    "action_default_subtract_false": {"actions": [{"action.arm": {"origin_keys": "action"}}]},
    "target_order": {"states": [{"observation.state.b": {"origin_keys": "sensor_b"}},
                                {"observation.state.a": {"origin_keys": "observation.state"}}],
                     "actions": [{"action.b": {"origin_keys": "sensor_b"}},
                                 {"action.a": {"origin_keys": "action"}}]},
}


@pytest.mark.parametrize("case", CASES)
def test_official_mapping_and_tensor_conformance(tmp_path, upstream, case):
    import torch
    robot = CASES[case]
    ours, path = adapter(tmp_path, robot)
    reference = official(upstream, path, robot)
    schema = ours.robot_schema
    for section in ("states", "actions", "images"):
        assert [f.target_name for f in getattr(schema, section)] == getattr(reference, section)
    for feature in schema.actions:
        assert feature.subtract_state == reference.action_subtract_state[feature.target_name]
        original = next(row[feature.target_name] for row in robot["actions"] if feature.target_name in row)
        assert feature.convert_from_state == original.get("convert_from_state", False)
    values = sample()
    untouched = copy.deepcopy(values)
    expected = reference.convert_features({k: torch.tensor(v) for k, v in values.items()})
    actual = schema.convert_features(values)
    assert set(actual) == set(expected)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key].numpy())
    frame = pd.DataFrame({k: list(v) for k, v in values.items() if FEATURES[k]["dtype"] != "video"})
    for normalized, canonical in zip(schema.states + schema.actions,
                                      ours.get_state_features() + ours.get_action_features()):
        assert canonical.width == normalized.dimension == actual[normalized.target_name].shape[-1]
        assert [(s.column, s.start, s.end) for s in canonical.slices] == [
            (s.origin_key, s.start, s.end) for s in normalized.sources]
        assert canonical.convert_from_state == normalized.convert_from_state
        np.testing.assert_array_equal(canonical.extract(frame), actual[normalized.target_name])
        mapping = reference.key_mapping.get(normalized.target_name)
        if mapping is None:
            assert normalized.direct
        elif isinstance(mapping["origin_keys"], str):
            assert normalized.sources[0].origin_key == mapping["origin_keys"]
        else:
            assert [s.origin_key for s in normalized.sources] == [k.split("*")[0] for k in mapping["origin_keys"]]
            assert [(s.declared_start, s.declared_end) for s in normalized.sources] == [
                (v["start"], v["end"]) for v in mapping["origin_keys"].values()]
    for image in schema.images:
        assert image.direct == (image.target_name in reference.feature_to_keep)
        canonical = ours.get_camera_features()[schema.images.index(image)]
        assert canonical.camera_column == image.origin_key
        if image.sources:
            assert canonical.width == image.dimension == actual[image.target_name].shape[-1]
            assert [(s.column, s.start, s.end) for s in canonical.slices] == [
                (s.origin_key, s.start, s.end) for s in image.sources]
    if schema.actions:
        tensors = {k: torch.tensor(v) for k, v in values.items()}
        tensors[reference.org_features["actions"][0] + "_is_pad"] = torch.zeros(3, dtype=torch.bool)
        expected_training = reference.apply(tensors)
        actual_training = schema.convert_features(values, subtract_state=True)
        for key in actual_training:
            np.testing.assert_allclose(actual_training[key], expected_training[key].numpy())
    for key in values:
        np.testing.assert_array_equal(values[key], untouched[key])


def test_convert_flag_without_origin_is_an_upstream_noop(tmp_path, upstream):
    robot = {"states": ["observation.state.arm"],
             "actions": [{"action.arm": {"convert_from_state": True}}]}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(robot), encoding="utf-8")
    reference = official(upstream, path, robot)
    import torch
    output = reference.convert_features({k: torch.tensor(v) for k, v in sample().items()})
    assert "action.arm" in reference.actions
    assert "action.arm" not in output  # No automatic state -> action conversion exists.
    with pytest.raises(ValueError, match="does not generate an action from state"):
        adapter(tmp_path, robot)


@pytest.mark.parametrize("robot, field", [
    ({"states": [{"state": {"origin_keys": ["observation.state"]}}]}, "origin_keys"),
    ({"images": [{"observation.images.top": {"origin_key": "observation.images.top"}}]}, "fields"),
    ({"actions": ["action.arm"]}, "entry"),
    ({"states": [{"state": {"origin_keys": "observation.state", "convert_from_state": True}}]}, "fields"),
    ({"states": [{"state": {"origin_keys": "missing"}}]}, "origin_keys"),
    ({"states": [{"state": {"origin_keys": [{"observation.state": {"start": 2, "end": 2}}]}}]}, "slice"),
    ({"states": [{"state": {"origin_keys": [{"observation.state": {"start": 0, "end": 2, "step": 1}}]}}]}, "slice"),
    ({"actions": [{"action.arm": {"origin_keys": "action", "subtract_state": True}}]}, "subtract_state"),
    ({"actions": [{"action.arm": {"origin_keys": "action", "subtract_state": "false"}}]}, "subtract_state"),
    ({"states": ["observation.state.arm", "observation.state.arm"]}, "target"),
    ({"states": [{"observation.state.arm": {"origin_keys": "sensor_b"}}],
      "actions": [{"action.arm": {"origin_keys": "action", "subtract_state": True}}]}, "subtract_state"),
    ({"actions": [{"action.effector.position": {"origin_keys": "action", "subtract_state": True}}]}, "subtract_state"),
])
def test_contextual_invalid_or_ambiguous_config(tmp_path, robot, field):
    with pytest.raises(ValueError, match="Invalid LingBot robot config:.*" + field):
        adapter(tmp_path, robot)


def test_reject_duplicate_yaml_keys(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("states: []\nstates: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        LingBotRobotConfigSchema.from_yaml(path, FEATURES)


def test_image_storage_and_empty_sections(tmp_path):
    features = copy.deepcopy(FEATURES)
    features["observation.images.top"]["dtype"] = "image"
    schema = LingBotRobotConfigSchema.parse({"states": [], "actions": [], "images": ["observation.images.top"]}, features)
    assert schema.images[0].canonical().camera_column == "observation.images.top"


def test_existing_droid_yaml_preserves_slices():
    path = PROJECT / "configs/robot_configs/droid_franka.yaml"
    features = {"observation.state": {"dtype": "float32", "shape": [8]},
                "action": {"dtype": "float32", "shape": [8]}}
    robot = yaml.safe_load(path.read_text(encoding="utf-8"))
    for row in robot["images"]:
        source = next(iter(row.values()))["origin_keys"]
        features[source] = {"dtype": "video", "shape": [32, 32, 3]}
    schema = LingBotRobotConfigSchema.parse(robot, features)
    assert [f.dimension for f in schema.states] == [7, 1]
    assert [f.dimension for f in schema.actions] == [7, 1]
    assert not any(f.convert_from_state for f in schema.actions)


@pytest.mark.parametrize("robot, expected_exception", [
    ({"states": [{"state": {"origin_keys": ["observation.state"]}}]}, AttributeError),
    ({"states": [{"state": {"origin_keys": [{"observation.state": {"start": None, "end": None}}]}}]}, TypeError),
    ({"actions": ["action.arm"]}, AssertionError),
    ({"actions": [{"action.arm": {}}]}, TypeError),
])
def test_official_rejects_apparent_shorthands(tmp_path, upstream, robot, expected_exception):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(robot), encoding="utf-8")
    with pytest.raises(expected_exception):
        official(upstream, path, robot)
    with pytest.raises(ValueError, match="Invalid LingBot robot config"):
        adapter(tmp_path, robot)


def test_singular_origin_key_is_not_a_rename(tmp_path, upstream):
    robot = {"images": [{"observation.images.renamed": {"origin_key": "observation.images.top"}}]}
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(robot), encoding="utf-8")
    reference = official(upstream, path, robot)
    assert "observation.images.renamed" not in reference.convert_features(sample())
    with pytest.raises(ValueError, match="origin_keys"):
        adapter(tmp_path, robot)


def test_yaml_order_is_distinct_from_training_padding_order(tmp_path, upstream):
    import torch
    robot = CASES["target_order"]
    ours, path = adapter(tmp_path, robot)
    reference = official(upstream, path, robot)
    reference.feature_config.joints = ["a", "b"]
    reference.feature_config.joints_max_dim = {"a": 6, "b": 4}
    values = sample()
    item = reference.convert_features({k: torch.tensor(v) for k, v in values.items()})
    for state in reference.states:
        item[state] = item[state][0]
    item.update(action_is_pad=torch.zeros(3, dtype=torch.bool), task="test")
    padded = reference.pad_and_concat(item)
    normalized = ours.robot_schema.convert_features(values)
    np.testing.assert_array_equal(padded["state"].numpy(), np.concatenate([
        normalized["observation.state.a"][0], normalized["observation.state.b"][0]]))
    np.testing.assert_array_equal(padded["action"].numpy(), np.concatenate([
        normalized["action.a"], normalized["action.b"]], axis=-1))
    assert [f.name for f in ours.get_state_features()] == ["observation.state.b", "observation.state.a"]


@pytest.mark.parametrize("convert", [False, True])
def test_episode_uses_explicit_action_source_without_applying_delta(tmp_path, convert):
    robot = {"states": [{"observation.state.arm": {"origin_keys": "observation.state"}}],
             "actions": [{"action.arm": {"origin_keys": "observation.state", "convert_from_state": convert,
                                         "subtract_state": True}}]}
    ours, _ = adapter(tmp_path, robot)
    values = sample()
    frame = pd.DataFrame({k: list(v) for k, v in values.items() if FEATURES[k]["dtype"] != "video"})
    frame["episode_index"] = 0
    frame["timestamp"] = np.arange(3) / 10
    view = ours.from_frame(frame).to_trajectory()
    np.testing.assert_array_equal(view.action, values["observation.state"])
    assert not view.action.flags.writeable
    np.testing.assert_array_equal(ours.robot_schema.convert_features(values, subtract_state=True)["action.arm"],
                                  np.zeros_like(values["observation.state"]))


def test_yaml_merge_override_matches_safe_loader(tmp_path, upstream):
    path = tmp_path / "merge.yaml"
    path.write_text("""states:
  - observation.state.a: &base
      origin_keys: observation.state
  - observation.state.b:
      <<: *base
      origin_keys: sensor_b
""", encoding="utf-8")
    schema = LingBotRobotConfigSchema.from_yaml(path, FEATURES)
    robot = yaml.safe_load(path.read_text(encoding="utf-8"))
    reference = official(upstream, path, robot)
    import torch
    expected = reference.convert_features({k: torch.tensor(v) for k, v in sample().items()})
    actual = schema.convert_features(sample())
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key].numpy())


def test_other_adapters_keep_their_canonical_contract(tmp_path):
    import json
    from lerobot_cleaner.adapters.lerobot_v3 import LeRobotAdapter
    from lerobot_cleaner.adapters.groot import GrootAdapter
    storage = SimpleNamespace(root=tmp_path, info={"features": FEATURES, "fps": 10})
    plain = LeRobotAdapter(tmp_path, storage=storage)
    assert plain.get_feature_schema().states[0].slices[0].column == "observation.state"
    assert plain.get_feature_schema().actions[0].slices[0].column == "action"
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/modality.json").write_text(json.dumps({
        "state": {"arm": {"start": 0, "end": 3}},
        "action": {"arm": {"start": 0, "end": 3}}, "video": {}}))
    groot = GrootAdapter(tmp_path, storage=storage)
    assert groot.get_feature_schema().raw_vectors
    assert groot.get_feature_schema().actions[0].width == 3
    assert not groot.get_feature_schema().actions[0].convert_from_state


def test_shared_mutating_yaml_mapping_is_rejected(tmp_path, upstream):
    path = tmp_path / "shared.yaml"
    path.write_text("""states:
  - observation.state.a: &shared
      origin_keys:
        - observation.state: {start: 0, end: 2}
  - observation.state.b: *shared
""", encoding="utf-8")
    robot = yaml.safe_load(path.read_text(encoding="utf-8"))
    reference = official(upstream, path, robot)
    import torch
    output = reference.convert_features({k: torch.tensor(v) for k, v in sample().items()})
    assert "observation.state.a" in output and "observation.state.b" not in output
    with pytest.raises(ValueError, match="shared mapping aliases"):
        LingBotRobotConfigSchema.from_yaml(path, FEATURES)
