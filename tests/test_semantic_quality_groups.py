"""Semantic quality selections, stable canonical ordering and legacy compatibility."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from lerobot_cleaner.adapters.schema import CanonicalFeatureSchema, FeatureSchema, FeatureSlice
from lerobot_cleaner.adapters.resolver import FeatureResolver
from lerobot_cleaner.adapters.builder import EpisodeBuilder
from lerobot_cleaner.v30.quality import QualityGroup, TrajectoryQualityConfig, audit_trajectory
from lerobot_cleaner.v30.quality_groups import QualityGroupResolver
from lerobot_cleaner.v30.calibration import CalibrationConfig, estimate_groups
from lerobot_cleaner.v30.v3 import V3Config


def schema(reordered=False):
    left = FeatureSchema("state.left", "state", (FeatureSlice("q", 0, 2),))
    grip = FeatureSchema("state.gripper", "state", (FeatureSlice("q", 2, 3),))
    right = FeatureSchema("state.right", "state", (FeatureSlice("q", 3, 5),))
    return CanonicalFeatureSchema(states=(grip, right, left) if reordered else (left, grip, right),
        actions=(FeatureSchema("action.arm", "action", (FeatureSlice("u", 0, 2),)),),
        cameras=(FeatureSchema("camera.front", "video", camera_column="cam"),))


def frame():
    q = np.zeros((8,5))
    q[4:,3:] = [3., 5.]
    return pd.DataFrame({"episode_index": [0]*8, "timestamp": np.arange(8)/10,
                         "q": list(q), "u": list(np.zeros((8,2)))})


def config(groups):
    return TrajectoryQualityConfig(groups=groups, timestamp={"enabled": False}, episode_structure=False)


def test_multiple_features_follow_schema_not_request_order_and_keep_provenance():
    cfg = config([{"name":"arms", "features":["state.right", "state.left"], "velocity": 1}])
    row = audit_trajectory(frame(),10,cfg,schema=schema())["groups"]["arms"]
    assert row["requested_features"] == ["state.right", "state.left"]
    assert row["resolved"]["feature_order"] == ["state.left", "state.right"]
    assert row["columns"] == [0,1,3,4] and row["resolved"]["dimension"] == 4
    assert row["resolved"]["feature_ranges"][1]["sources"] == [{"column":"q", "start":3, "end":5}]
    assert not row["checks"]["velocity"]["passed"]
    assert "features" not in row["thresholds"]


def test_same_group_tracks_changed_layout_without_rewriting_numbers():
    cfg = config([{"name":"right", "feature":"state.right", "velocity":1}])
    first = audit_trajectory(frame(),10,cfg,schema=schema())["groups"]["right"]
    shifted = audit_trajectory(frame(),10,cfg,schema=schema(True))["groups"]["right"]
    assert first["columns"] == [3,4] and shifted["columns"] == [1,2]
    assert first["checks"] == shifted["checks"]
    assert cfg.groups[0].columns is None and cfg.groups[0].source is None


def test_semantic_and_legacy_produce_identical_checks_and_legacy_order_survives():
    cfg = config([{"name":"semantic", "feature":"state.right", "velocity":1},
                  {"name":"legacy", "source":"state", "columns":[3,4], "velocity":1},
                  {"name":"reverse", "source":"state", "columns":[4,3]}])
    rows = audit_trajectory(frame(),10,cfg,schema=schema())["groups"]
    assert rows["semantic"]["checks"] == rows["legacy"]["checks"]
    assert rows["reverse"]["columns"] == [4,3]
    assert rows["legacy"]["selection_mode"] == "legacy"
    assert rows["semantic"]["selection_mode"] == "semantic"


@pytest.mark.parametrize("fields", [{"feature":"state.right"}, {"features":["state.left","state.right"]},
    {"source":{"feature":"state.right"}}, {"source":{"features":["state.left"]}},
    {"source":"action", "columns":[1,0]}])
def test_three_selector_forms_and_nested_alias_roundtrip(fields):
    group = QualityGroup(name="arms", **fields)
    cfg = config([group])
    assert TrajectoryQualityConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert QualityGroup.model_validate(group.model_dump()) == group


@pytest.mark.parametrize("fields", [
    {}, {"feature":"x", "features":["y"]}, {"feature":"x", "source":"state"},
    {"features":["x"], "columns":[0]}, {"source":"state"}, {"columns":[0]},
    {"features":[]}, {"features":{"x","y"}}, {"features":["x","x"]}, {"feature":" "},
    {"source":{"feature":"x"}, "feature":"y"}, {"source":{"feature":"x", "columns":[0]}},
])
def test_ambiguous_empty_or_duplicate_selectors_fail_early(fields):
    with pytest.raises(ValidationError):
        QualityGroup(name="arms", **fields)


@pytest.mark.parametrize("features,feature,reason", [
    (["state.missing"],"state.missing","does not exist"),
    (["camera.front"],"camera.front","allowed numeric modality"),
    (["state.left","action.arm"],"action.arm","cannot combine"),
    (["language"],"language","not a state/action feature"),
])
def test_resolution_errors_identify_group_feature_and_reason(features,feature,reason):
    group = QualityGroup(name="all_arms", features=features)
    with pytest.raises(ValueError) as caught:
        QualityGroupResolver().resolve(group, schema(), {"state":5,"action":2})
    for part in ("all_arms",feature,reason):
        assert part in str(caught.value)


def test_empty_dimension_and_runtime_layout_mismatch():
    empty = CanonicalFeatureSchema(states=(FeatureSchema("state.empty","state"),))
    with pytest.raises(ValueError,match="arms.*state.empty.*dimension"):
        QualityGroupResolver().resolve(QualityGroup(name="arms", feature="state.empty"),empty,{"state":0,"action":0})
    with pytest.raises(ValueError,match="arms.*width 4"):
        QualityGroupResolver().resolve(QualityGroup(name="arms",feature="state.left"),schema(),{"state":4,"action":2})


def test_raw_vector_schema_uses_source_slices_including_gaps():
    raw = CanonicalFeatureSchema(raw_vectors=True, states=(
        FeatureSchema("state.left","state",(FeatureSlice("observation.state",2,4),)),
        FeatureSchema("state.right","state",(FeatureSlice("observation.state",6,8),))))
    result = FeatureResolver().resolve_selection(raw,["state.right","state.left"],widths={"state":9,"action":0})
    assert result["columns"] == [2,3,6,7] and result["dimension"] == 4
    overlap = CanonicalFeatureSchema(raw_vectors=True, states=raw.states+(
        FeatureSchema("state.alias","state",(FeatureSlice("observation.state",3,4),)),))
    with pytest.raises(ValueError,match="state.alias.*overlaps"):
        FeatureResolver().resolve_selection(overlap,["state.left","state.alias"],widths={"state":9,"action":0})


def test_frame_only_bridge_only_exposes_whole_vector_names():
    data = frame().rename(columns={"q":"observation.state", "u":"action"})
    result = audit_trajectory(data,10,config([{"name":"whole", "feature":"observation.state"}]))
    assert result["groups"]["whole"]["resolved"]["dimension"] == 5
    # No inferred arm names merely because this vector has a familiar width.
    with pytest.raises(ValueError,match="arm.*state.arm.*does not exist"):
        audit_trajectory(data,10,config([{"name":"arm", "feature":"state.arm"}]))


def test_adapter_supplies_schema_and_checks_do_not_mutate_frames():
    data = frame()
    before = data.copy(deep=True)
    builder = EpisodeBuilder(schema(),10)
    adapter = SimpleNamespace(from_frame=builder.build)
    result = audit_trajectory(data,10,config([{"name":"right","feature":"state.right"}]),adapter=adapter)
    assert result["groups"]["right"]["source"] == "state"
    pd.testing.assert_frame_equal(data,before)


def test_calibration_keeps_semantic_selectors_and_records_resolved_layout():
    cfg = config([{"name":"right", "feature":"state.right"}])
    records = [audit_trajectory(frame(),10,cfg,schema=schema()) for _ in range(2)]
    groups, ready = estimate_groups(records,cfg,CalibrationConfig(min_samples=2))
    assert ready and groups["right"]["source"] == "state"
    assert groups["right"]["columns"] == [3,4]
    assert groups["right"]["requested_features"] == ["state.right"]
    raw = cfg.model_dump()
    for key,value in groups["right"]["metrics"].items():
        raw["groups"][0][key] = value["threshold"]
    generated = TrajectoryQualityConfig.model_validate(raw)
    assert generated.groups[0].feature == "state.right" and generated.groups[0].columns is None
    records[1] = audit_trajectory(frame(),10,cfg,schema=schema(True))
    with pytest.raises(ValueError,match="right.*changed layout"):
        estimate_groups(records,cfg,CalibrationConfig(min_samples=2))


def test_shipped_libero_semantic_config_matches_legacy_quality():
    from lerobot_cleaner.adapters.generic import GenericMappingAdapter
    project = Path(__file__).resolve().parents[1]
    legacy = V3Config.from_yaml(project/"configs/cleaning/libero_v3.yaml")
    semantic = V3Config.from_yaml(project/"configs/cleaning/libero_v3_semantic.yaml")
    storage = SimpleNamespace(root=project, info={"fps":10,"features":{
        "observation.state":{"dtype":"float32","shape":[8]},"action":{"dtype":"float32","shape":[7]}}})
    adapter = GenericMappingAdapter(project,semantic.mapping_config,semantic,storage=storage)
    data = pd.DataFrame({"episode_index":[0]*8,"timestamp":np.arange(8)/10,
                         "observation.state":list(np.zeros((8,8))),"action":list(np.zeros((8,7)))})
    old = audit_trajectory(data,10,legacy.quality)
    new = audit_trajectory(data,10,semantic.quality,adapter=adapter)
    for name,group in old["groups"].items():
        assert group["checks"] == new["groups"][name]["checks"]
        assert group["columns"] == new["groups"][name]["columns"]


def test_droid_migrated_groups_resolve_to_the_original_layout():
    from lerobot_cleaner.adapters.lingbot import LingBotAdapter
    project = Path(__file__).resolve().parents[1]
    cfg = V3Config.from_yaml(project/"configs/cleaning/droid_v3.yaml")
    features = {"observation.state":{"dtype":"float32","shape":[8]},
                "action":{"dtype":"float32","shape":[8]}}
    features.update({"observation.images."+name:{"dtype":"video","shape":[24,32,3]}
                     for name in ("exterior_1_left","exterior_2_left","wrist_left")})
    storage = SimpleNamespace(root=project,info={"fps":10,"features":features})
    adapter = LingBotAdapter(project,cfg.robot_config,cfg,storage=storage)
    resolved = [QualityGroupResolver().resolve(group,adapter.get_feature_schema(),{"state":8,"action":8})["resolved"]
                for group in cfg.quality.groups]
    assert [(row["source"],row["columns"]) for row in resolved] == [
        ("state",list(range(7))),("action",list(range(7))),("state",[7]),("action",[7])]
