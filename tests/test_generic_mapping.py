"""Generic mapping, adapter integration and model-independent import boundary."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml

from lerobot_cleaner.adapters.generic import GenericMappingAdapter
from lerobot_cleaner.adapters.lerobot_v3 import LeRobotAdapter
from lerobot_cleaner.adapters.factory import v3_adapter
from lerobot_cleaner.adapters.mapping import FeatureSpec, SourceSpec, load_mapping, parse_mapping
from lerobot_cleaner.adapters.resolver import FeatureResolver
from lerobot_cleaner.v30.v3 import V3Config


FEATURES = {key: {"dtype": "float32", "shape": [width]} for key, width in {
    "left_arm.position": 3, "right_arm.position": 2, "observation.gripper": 1,
    "action.arm": 4, "action.gripper": 1}.items()}
MAPPING = {"canonical": {
    "state": {"arm": [{"source": "right_arm.position"}, {"source": "left_arm.position", "start": 1, "end": 3}],
              "gripper": [{"source": "observation.gripper"}]},
    "action": {"arm": [{"source": "action.arm"}], "gripper": [{"source": "action.gripper"}]}}}


def inputs(tmp_path, mapping=None):
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump(mapping or MAPPING, sort_keys=False), encoding="utf-8")
    info = {"fps": 10, "features": FEATURES, "total_tasks": 1, "total_episodes": 1, "total_frames": 3}
    storage = SimpleNamespace(root=tmp_path, info=info,
        tasks=pd.DataFrame({"task_index": [0]}, index=["pick cup"]))
    frame = pd.DataFrame({"episode_index": [0]*3, "task_index": [0]*3,
        "timestamp": [0., .1, .2], "index": [0,1,2], "frame_index": [0,1,2],
        **{key: [np.arange(spec["shape"][0])+i*10 for i in range(3)] for key, spec in FEATURES.items()}})
    return path, storage, frame


def test_concat_slice_order_named_features_and_trajectory_compatibility(tmp_path):
    path, storage, frame = inputs(tmp_path)
    adapter = GenericMappingAdapter(tmp_path, path, storage=storage)
    schema = adapter.get_feature_schema()
    assert [(f.name, f.width) for f in schema.states] == [("state.arm", 4), ("state.gripper", 1)]
    assert [(f.name, f.width) for f in schema.actions] == [("action.arm", 4), ("action.gripper", 1)]
    episode = adapter.from_frame(frame)
    named = episode.feature_arrays("state")
    np.testing.assert_array_equal(named["state.arm"][0], [0, 1, 1, 2])
    assert not named["state.arm"].flags.writeable
    np.testing.assert_array_equal(episode.to_trajectory().state[0], [0, 1, 1, 2, 0])
    assert episode.language.samples[0].task_text == "pick cup"
    assert "lingbot" not in adapter.describe()


@pytest.mark.parametrize("entry,width", [
    ("left_arm.position", 3), ({"source": "left_arm.position"}, 3),
    ([{"source": "left_arm.position", "start": 1}], 2),
    ([{"source": "left_arm.position", "end": 2}], 2),
    ({"sources": [{"source": "left_arm.position"}], "dimension": 3}, 3),
])
def test_direct_mapping_and_dimension_inference(entry, width):
    specs = parse_mapping({"canonical": {"state": {"renamed": entry}}})
    result = FeatureResolver().resolve_mapping(specs, FEATURES)
    assert result[0].name == "state.renamed" and result[0].width == width


@pytest.mark.parametrize("source", [
    SourceSpec("missing"), SourceSpec("left_arm.position", -1, 2),
    SourceSpec("left_arm.position", 0, 4), SourceSpec("left_arm.position", 1, 1),
    SourceSpec("left_arm.position", 2, 1), SourceSpec("left_arm.position", True, 2),
    SourceSpec("left_arm.position", 0, 2.0), SourceSpec(123),
])
def test_invalid_sources_have_full_error_context(source):
    with pytest.raises(ValueError) as caught:
        FeatureResolver().resolve_mapping((FeatureSpec("state.arm", "state", (source,)),), FEATURES)
    message = str(caught.value)
    for part in ("target=", "state.arm", "source=", "slice=", "actual dimension="):
        assert part in message


def test_concat_dimension_and_target_conflicts():
    resolver = FeatureResolver()
    spec = FeatureSpec("state.arm", "state", (SourceSpec("left_arm.position"),), 4)
    with pytest.raises(ValueError, match="expected 4"):
        resolver.resolve_mapping((spec,), FEATURES)
    spec = FeatureSpec("state.arm", "state", (SourceSpec("left_arm.position"),))
    with pytest.raises(ValueError, match="conflicting target"):
        resolver.resolve_mapping((spec, spec), FEATURES)
    with pytest.raises(ValueError, match="ordered sequence"):
        resolver.resolve_mapping((FeatureSpec("state.arm", "state", set()),), FEATURES)
    alias = parse_mapping({"canonical": {"state": {"arm": "left_arm.position", "state.arm": "right_arm.position"}}})
    with pytest.raises(ValueError, match="conflicting target"):
        resolver.resolve_mapping(alias, FEATURES)


@pytest.mark.parametrize("spec", [{"dtype": "string", "shape": [3]},
    {"dtype": "float32", "shape": [2, 3]}, {"dtype": "float32", "shape": [0]}])
def test_source_metadata_must_be_numeric_vector(spec):
    with pytest.raises(ValueError, match="numeric vector"):
        FeatureResolver().resolve_mapping((FeatureSpec("state.arm", "state", (SourceSpec("q"),)),), {"q": spec})


def test_yaml_rejects_duplicate_keys_unknown_fields_and_unordered_sources(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("canonical:\n  state:\n    arm: q\n    arm: u\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_mapping(path)
    for entry in ([], {"q": {}, "u": {}}, [{"source": "q", "step": 2}]):
        with pytest.raises(ValueError):
            parse_mapping({"canonical": {"state": {"arm": entry}}})


def test_runtime_dimension_changes_are_not_hidden_by_a_valid_slice(tmp_path):
    path, storage, frame = inputs(tmp_path)
    adapter = GenericMappingAdapter(tmp_path, path, storage=storage)
    for i in range(len(frame)):
        frame.at[i, "left_arm.position"] = np.arange(4)
    with pytest.raises(ValueError, match="actual dimension=4.*metadata 3"):
        adapter.from_frame(frame).to_trajectory()


def test_default_adapter_retains_existing_names_and_does_not_guess(tmp_path):
    _, storage, frame = inputs(tmp_path)
    adapter = LeRobotAdapter(tmp_path, storage=storage)
    assert adapter.get_state_features() == adapter.get_action_features() == ()
    storage.info = {**storage.info, "features": {"observation.state": {"dtype": "float32", "shape": [3]},
                                               "action": {"dtype": "float32", "shape": [4]}}}
    adapter = LeRobotAdapter(tmp_path, storage=storage)
    assert [(f.name, f.width) for f in adapter.get_state_features()] == [("observation.state", 3)]
    assert [(f.name, f.width) for f in adapter.get_action_features()] == [("action", 4)]


def test_factory_and_yaml_relative_paths(tmp_path):
    path, storage, _ = inputs(tmp_path)
    config_path = tmp_path / "clean.yaml"
    config_path.write_text("mapping_config: mapping.yaml\n", encoding="utf-8")
    config = V3Config.from_yaml(config_path)
    assert config.mapping_config == path.resolve()
    assert isinstance(v3_adapter(tmp_path, config, storage=storage), GenericMappingAdapter)
    with pytest.raises(ValueError, match="mapping_config"):
        V3Config(semantic_adapter="generic")
    with pytest.raises(ValueError, match="multiple mappings"):
        V3Config(mapping_config=path, robot_config=path)
    with pytest.raises(ValueError, match="requires semantic_adapter"):
        V3Config(semantic_adapter="lerobot", mapping_config=path)


def test_v3_audit_entry_uses_generic_mapping(tmp_path, monkeypatch):
    from lerobot_cleaner.v30 import pipeline
    path, storage, frame = inputs(tmp_path)
    class Storage:
        def __init__(self, *args): self.__dict__.update(storage.__dict__)
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def episode(self, eid):
            return SimpleNamespace(to_pandas=lambda: frame.copy()), {"episode_index": 0, "length": 3}
        def validate_episode(self, *args): pass
        def video_references(self, eid): return []
        def output_layout(self): return [], [], {}
        peak_rows = 3
    monkeypatch.setattr(pipeline, "OfficialStorage", Storage)
    monkeypatch.setattr(pipeline, "prepare_dataset", lambda dataset, config: dataset)
    seen = []
    observer = SimpleNamespace(set_feature_schema=lambda schema: seen.append(schema),
        process=lambda metadata, frame: None, result=lambda: {})
    report = pipeline.audit_pipeline(tmp_path, V3Config(mapping_config=path, quality={"video": {"enabled": False},
        "groups": [{"name": "arm", "feature": "state.arm"}]}), observer=observer)
    assert seen[0].states[0].name == "state.arm"
    assert report["adapter"]["type"] == "GenericMappingAdapter"
    assert report["trajectory_quality"][0]["groups"]["arm"]["resolved"]["columns"] == [0, 1, 2, 3]
    assert report["trajectory_quality"][0]["checks"]["finite"]["passed"]
    assert report["dataset_quality"]["language"]["episodes_with_task"] == 1


def test_generic_import_does_not_load_model_integrations():
    code = """
import sys
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.startswith(('lerobot_cleaner.adapters.lingbot', 'lingbotvla', 'transformers')):
            raise ImportError('Model integration must not load: ' + fullname)
sys.meta_path.insert(0, Block())
from lerobot_cleaner.adapters import GenericMappingAdapter, FeatureResolver
from lerobot_cleaner.v30.v3 import V3Config
from lerobot_cleaner.adapters.mapping import parse_mapping
mapping = parse_mapping({'canonical': {'state': {'arm': 'q'}}})
assert FeatureResolver().resolve_mapping(mapping, {'q': {'dtype':'float32', 'shape':[2]}})[0].width == 2
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
