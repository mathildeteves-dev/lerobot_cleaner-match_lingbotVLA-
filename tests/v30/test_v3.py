"""Native v3 tests do not need the large private sample or a video decoder."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from lerobot_cleaner.cli import app
from lerobot_cleaner.v30.lingbot.config import validate_mapping
from lerobot_cleaner.v30.v3 import Alias, V3Config, audit_v3, clean_v3, load_v3

PROJECT = Path(__file__).parents[2]


def alias_config(**kwargs):
    return V3Config(
        aliases=[
            Alias(source="action", target="action.joint_position", start=0, end=7),
            Alias(source="action", target="action.gripper_position", start=7, end=8),
        ],
        **kwargs,
    )


def mutate_actions(root, mutate):
    path = root / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(path)
    actions = np.stack(data.action).copy()
    mutate(actions)
    data["action"] = list(actions)
    data["action.joint_position"] = list(actions[:, :7])
    data["action.gripper_position"] = actions[:, 7]
    data.to_parquet(path, index=False)


def test_audit(v3_data):
    report = audit_v3(v3_data)
    assert report["frames"] == 6
    assert report["unsuccessful_episodes"] == 1
    assert report["video_verification"] == "metadata_only"
    assert len(report["videos"]) == 3


def test_clean_preserves_rows_videos_metadata_and_extra_columns(v3_data, tmp_path):
    original = {p.relative_to(v3_data): p.read_bytes() for p in v3_data.rglob("*") if p.is_file()}
    output = tmp_path / "clean"
    report = clean_v3(v3_data, output, alias_config())
    assert report["changed_values"] == {}
    assert report["rows_removed"] == 0
    _, _, data, episodes, _, _, _ = load_v3(output)
    assert len(data) == 6 and len(episodes) == 2
    assert data.extra_unlisted_column.tolist() == ["preserve me"] * 6
    assert episodes["videos/observation.images.wrist_left/from_timestamp"].tolist() == [0.0, 0.2]
    assert data["action.gripper_position"].dtype == np.float32
    for relative, contents in original.items():
        assert (v3_data / relative).read_bytes() == contents
        if str(relative).startswith("videos"):
            assert (output / relative).read_bytes() == contents
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [6]
    np.testing.assert_allclose(
        stats["action"]["mean"], np.stack(data.action).mean(axis=0), rtol=1e-6
    )
    assert stats["observation.images.wrist_left"]["mean"] == [[[0.5]], [[0.5]], [[0.5]]]
    assert (output / "cleaning_report/report.json").is_file()


def test_nonfinite_default_fails_without_output(v3_data, tmp_path):
    mutate_actions(v3_data, lambda a: a.__setitem__((1, 0), np.nan))
    with pytest.raises(ValueError, match="Non-finite"):
        clean_v3(v3_data, tmp_path / "clean", alias_config())
    assert not (tmp_path / "clean").exists()


def test_interpolation_is_within_episode_and_updates_aliases(v3_data, tmp_path):
    mutate_actions(v3_data, lambda a: a.__setitem__((3, 0), np.nan))
    report = clean_v3(v3_data, tmp_path / "clean", alias_config(nonfinite="interpolate"))
    data = pd.read_parquet(tmp_path / "clean/data/chunk-000/file-000.parquet")
    assert data.action.iloc[3][0] == pytest.approx(
        0.32
    )  # first value from episode 1, not episode 0
    assert data["action.joint_position"].iloc[3][0] == pytest.approx(0.32)
    assert report["changed_values"]["action"] == 1


def test_entire_invalid_episode_fails(v3_data, tmp_path):
    mutate_actions(v3_data, lambda a: a.__setitem__((slice(0, 3), 0), np.nan))
    with pytest.raises(ValueError, match="entirely invalid"):
        clean_v3(v3_data, tmp_path / "clean", alias_config(nonfinite="interpolate"))


def test_explicit_clipping_synchronizes_aliases(v3_data, tmp_path):
    clean_v3(v3_data, tmp_path / "clean", alias_config(bounds={"action": (0, 0.2)}))
    data = pd.read_parquet(tmp_path / "clean/data/chunk-000/file-000.parquet")
    assert np.stack(data.action).max() == pytest.approx(0.2)
    np.testing.assert_array_equal(
        np.stack(data.action)[:, :7], np.stack(data["action.joint_position"])
    )
    np.testing.assert_array_equal(np.stack(data.action)[:, 7], data["action.gripper_position"])


@pytest.mark.parametrize("inside", [True, False])
def test_unsafe_output_rejected(v3_data, tmp_path, inside):
    output = v3_data / "child" if inside else tmp_path
    with pytest.raises(ValueError, match="new directory outside"):
        clean_v3(v3_data, output)


def test_alias_mismatch_fails(v3_data, tmp_path):
    path = v3_data / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(path)
    data["action.gripper_position"] = 999.0
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Alias mismatch"):
        clean_v3(v3_data, tmp_path / "clean", alias_config())


def test_missing_video_rejected(v3_data):
    next(v3_data.glob("videos/*/chunk-*/*.mp4")).unlink()
    with pytest.raises(ValueError, match="Missing/empty"):
        audit_v3(v3_data)


def test_video_path_traversal_rejected(v3_data):
    path = v3_data / "meta/info.json"
    info = json.loads(path.read_text())
    info["video_path"] = "../outside.mp4"
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes root"):
        audit_v3(v3_data)


def test_frame_indices_rejected(v3_data):
    path = v3_data / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(path)
    data.loc[1, "frame_index"] = 42
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="frame_index"):
        audit_v3(v3_data)


def test_config_rejects_unsupported_rules():
    with pytest.raises(ValueError):
        V3Config.model_validate({"drop_failed_episodes": True})


def test_mapping_matches_droid(v3_data):
    result = validate_mapping(
        v3_data,
        PROJECT / "configs/robot_configs/droid_franka.yaml",
        PROJECT / "configs/vla/droid_franka.yaml",
    )
    assert result["mapped_dimensions"]["action.arm.position"] == 7
    assert result["mapped_dimensions"]["action.effector.position"] == 1
    assert len(result["cameras"]) == 3


def test_r1pro_does_not_match_droid(v3_data):
    with pytest.raises(ValueError, match="Slice out of bounds"):
        validate_mapping(
            v3_data,
            PROJECT / "configs/robot_configs/r1pro.yaml",
            PROJECT / "configs/vla/r1pro_load20000h.yaml",
        )


def test_bad_camera_mapping_rejected(v3_data, tmp_path):
    robot = yaml.safe_load((PROJECT / "configs/robot_configs/droid_franka.yaml").read_text())
    robot["images"][0]["observation.images.camera_exterior_1"]["origin_keys"] = "missing"
    path = tmp_path / "droid_franka.yaml"
    path.write_text(yaml.safe_dump(robot), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing video"):
        validate_mapping(v3_data, path, PROJECT / "configs/vla/droid_franka.yaml")


def test_cli_v3_audit_and_clean(v3_data, tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["audit-v3", str(v3_data)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["clean-v3", str(v3_data), "-o", str(tmp_path / "clean")])
    assert result.exit_code == 0, result.output


def test_norm_launcher_uses_absolute_paths_and_correct_cwd_inputs(v3_data, tmp_path):
    from scripts.run_lingbot_norm import build_command

    lingbot = tmp_path / "lingbot"
    (lingbot / "scripts").mkdir(parents=True)
    (lingbot / "train.sh").touch()
    (lingbot / "scripts/compute_norm.py").touch()
    robot = PROJECT / "configs/robot_configs/droid_franka.yaml"
    train = PROJECT / "configs/vla/droid_franka.yaml"
    norm = tmp_path / "norm.json"
    command = build_command(lingbot, v3_data, robot, train, norm)
    assert command[:5] == ["bash", "-o", "pipefail", "train.sh", "scripts/compute_norm.py"]
    assert command[5] == str(train.resolve())
    assert command[command.index("--data.train_path") + 1] == str(v3_data.resolve())
    assert command[command.index("--data.robot_config_root") + 1] == str(robot.parent.resolve())
    assert command[-1] == str(norm.resolve())
    assert not norm.exists()


def test_norm_launcher_rejects_wrong_repository(v3_data, tmp_path):
    from scripts.run_lingbot_norm import build_command

    with pytest.raises(ValueError, match="Missing LingBot source"):
        build_command(
            tmp_path,
            v3_data,
            PROJECT / "configs/robot_configs/droid_franka.yaml",
            PROJECT / "configs/vla/droid_franka.yaml",
            tmp_path / "norm.json",
        )
