"""Unified adapter contracts; no LingBot runtime or video decoder needed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from lerobot_cleaner.adapters import DatasetAdapter, GrootAdapter, LingBotAdapter, UnifiedEpisode
from lerobot_cleaner.adapters.factory import v3_adapter
from lerobot_cleaner.v21.reader import LeRobotDataset
from lerobot_cleaner.v21.types import EpisodeWork
from lerobot_cleaner.v30.v3 import V3Config, audit_v3, clean_v3

PROJECT = Path(__file__).resolve().parents[1]
ROBOT = PROJECT / "configs/robot_configs/droid_franka.yaml"


def test_abstract_adapter_and_compatibility_exports():
    with pytest.raises(TypeError):
        DatasetAdapter()
    assert LeRobotDataset is GrootAdapter
    assert EpisodeWork is UnifiedEpisode


def test_groot_retains_modality_contract(tmp_path):
    root = tmp_path / "groot"
    (root / "meta").mkdir(parents=True)
    info = {"fps": 10, "chunks_size": 1000, "data_path": "episode_{episode_index:06d}.parquet",
            "features": {"observation.state": {"dtype": "float32", "shape": [3]},
                         "action": {"dtype": "float32", "shape": [3]}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(FileNotFoundError, match="modality"):
        GrootAdapter(root)
    modality = {"state": {"arm": {"start": 0, "end": 2}, "gripper": {"start": 2, "end": 3}},
                "action": {"arm": {"start": 0, "end": 2}, "gripper": {"start": 2, "end": 3}}, "video": {}}
    (root / "meta/modality.json").write_text(json.dumps(modality))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 2, "tasks": ["pick"]}))
    frame = pd.DataFrame({"observation.state": [[1., 2, 3], [2., 3, 4]], "action": [[4., 5, 6], [5., 6, 7]],
                          "timestamp": [0., .1], "episode_index": [0, 0]})
    frame.to_parquet(root / "episode_000000.parquet", index=False)
    adapter = GrootAdapter(root)
    episode = adapter.read_episode(0)
    assert isinstance(episode, UnifiedEpisode)
    assert adapter.get_state_features()[1].width == 1
    assert adapter.resolver.resolve("action.gripper").start == 2
    np.testing.assert_array_equal(episode.to_trajectory().state, [[1, 2, 3], [2, 3, 4]])
    with pytest.raises(ValueError, match="No output writer"):
        adapter.write_episode(episode)
    target = tmp_path / "staged.parquet"
    adapter.write_episode(episode, writer=lambda ep: ep.df.to_parquet(target))
    assert target.is_file()


def test_lingbot_reads_without_modality_and_streams(v3_data, monkeypatch):
    assert not (v3_data / "meta/modality.json").exists()
    adapter = LingBotAdapter(v3_data, ROBOT)
    def forbidden(*args, **kwargs):
        raise AssertionError("Adapter must not use full pandas parquet reads")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    episode = adapter.read_episode(1)
    assert episode.episode_index == 1 and len(episode.df) == 3
    assert episode.to_trajectory().state.shape == (3, 8)
    assert [feature.width for feature in adapter.get_action_features()] == [7, 1]
    assert len(adapter.get_camera_features()) == 3
    with pytest.raises(KeyError, match="Unknown episode"):
        adapter.read_episode(3)


@pytest.mark.parametrize("engine", ["memory", "streaming"])
def test_lingbot_mapping_drives_quality_and_row_preserving_clean(v3_data, tmp_path, engine):
    robot = yaml.safe_load(ROBOT.read_text(encoding="utf-8"))
    robot["states"] = [{"observation.state.arm.position": {"origin_keys": [
        {"observation.state": {"start": 7, "end": 8}},
        {"observation.state": {"start": 0, "end": 2}}]}}]
    robot["actions"] = [{"action.arm.position": {"origin_keys": [
        {"action.gripper_position": {"start": 0, "end": 1}},
        {"action.joint_position": {"start": 0, "end": 2}}], "subtract_state": False}}]
    path = tmp_path / "mapped.yaml"
    path.write_text(yaml.safe_dump(robot))
    cfg = V3Config(robot_config=path, engine=engine, disk_reserve_gb=0,
                   quality={"groups": [{"name": "arm", "source": "action", "columns": [0, 1, 2]}]})
    adapter = v3_adapter(v3_data, cfg)
    episode = adapter.read_episode(0)
    original = np.stack(episode.df.action)
    np.testing.assert_allclose(episode.to_trajectory().action, original[:, [7, 0, 1]])
    # Malformed GR00T metadata must be irrelevant to the LingBot parser.
    (v3_data / "meta/modality.json").write_text("not JSON")
    audited = audit_v3(v3_data, cfg)
    assert audited["adapter"]["type"] == "LingBotAdapter"
    assert audited["adapter"]["action_features"][0]["slices"][0]["column"] == "action.gripper_position"
    report = clean_v3(v3_data, tmp_path / "clean", cfg)
    assert report["adapter"] == audited["adapter"]
    assert report["trajectory_quality_input"] == report["trajectory_quality_output"]
    assert report["rows_removed"] == 0 and report["changed_values"] == {}
    output = pd.read_parquet(tmp_path / "clean/data/chunk-000/file-000.parquet")
    np.testing.assert_allclose(np.stack(output.action)[:3], original)


def test_v3_adapter_rejects_identity_changes(v3_data):
    adapter = LingBotAdapter(v3_data, ROBOT, writer=lambda ep: ep.df)
    episode = adapter.read_episode(0)
    episode.df.loc[0, "timestamp"] += 1
    with pytest.raises(ValueError, match="identity"):
        adapter.write_episode(episode)
    episode = adapter.read_episode(0)
    episode.restrict_to([0, 1])
    with pytest.raises(ValueError, match="dropping"):
        adapter.write_episode(episode)


def test_yaml_relative_robot_path_and_native_fallback(v3_data, tmp_path):
    robot = tmp_path / "robot.yaml"
    robot.write_bytes(ROBOT.read_bytes())
    cfg_file = tmp_path / "clean.yaml"
    cfg_file.write_text("robot_config: robot.yaml\n")
    cfg = V3Config.from_yaml(cfg_file)
    assert cfg.robot_config == robot.resolve()
    assert isinstance(v3_adapter(v3_data, cfg), LingBotAdapter)
    assert type(v3_adapter(v3_data, V3Config())).__name__ == "LeRobotV3Adapter"


def test_bad_robot_slice_fails_before_clean_output(v3_data, tmp_path):
    robot = yaml.safe_load(ROBOT.read_text(encoding="utf-8"))
    robot["actions"][0]["action.arm.position"]["origin_keys"][0]["action"]["end"] = 0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(robot))
    with pytest.raises(ValueError, match="empty slice"):
        clean_v3(v3_data, tmp_path / "out", V3Config(robot_config=path))
    assert not (tmp_path / "out").exists()


def test_groot_pipeline_stages_and_finalizes_without_video_dependency(tmp_path):
    from lerobot_cleaner.v21.config import CleaningConfig
    from lerobot_cleaner.v21.pipeline import Pipeline
    root = tmp_path / "groot"
    (root / "meta").mkdir(parents=True)
    info = {"codebase_version": "v2.1", "fps": 10, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "total_frames": 2, "total_episodes": 1, "total_tasks": 1,
            "features": {"observation.state": {"dtype": "float32", "shape": [3]},
                         "action": {"dtype": "float32", "shape": [3]}}}
    modality = {section: {"arm": {"start": 0, "end": 2}, "gripper": {"start": 2, "end": 3}}
                for section in ["state", "action"]}
    modality["video"] = {}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/modality.json").write_text(json.dumps(modality))
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 2, "tasks": ["pick"]}) + "\n")
    (root / "data/chunk-000").mkdir(parents=True)
    frame = pd.DataFrame({"observation.state": [[1., 2, .2], [2., 3, .8]],
                          "action": [[4., 5, .2], [5., 6, .8]], "timestamp": [0., .1],
                          "index": [0, 1], "frame_index": [0, 1], "episode_index": [0, 0], "task_index": [0, 0]})
    frame.to_parquet(root / "data/chunk-000/episode_000000.parquet", index=False)
    cfg = CleaningConfig(input=root, output=tmp_path / "out", num_workers=1)
    for name in type(cfg.rules).model_fields:
        getattr(cfg.rules, name).enabled = False
    cfg.rules.gripper_binarize.enabled = True
    cfg.rules.gripper_binarize.targets = ["state.gripper", "action.gripper"]
    pipeline = Pipeline(cfg)
    assert isinstance(pipeline.source, DatasetAdapter)
    report = pipeline.run()
    assert report.kept_episodes == 1
    result = GrootAdapter(tmp_path / "out").read_episode(0)
    np.testing.assert_array_equal(result.to_trajectory().action[:, 2], [0, 1])
    assert json.loads((tmp_path / "out/meta/modality.json").read_text()) == modality
    assert (tmp_path / "out/meta/stats.json").is_file()
