"""generate-lingbot-config: spec -> validated LingBot robot config + train config pair."""

import json
from pathlib import Path

import pytest
import yaml
from test_v3 import PROJECT

from lerobot_cleaner.cli import app
from lerobot_cleaner.v30.lingbot.config import validate_mapping
from lerobot_cleaner.v30.lingbot.generate import generate_configs

DROID_SPEC = PROJECT / "configs/embodiments/droid_franka.yaml"
R1PRO_SPEC = PROJECT / "configs/embodiments/r1pro.yaml"


def generate(v3_data, tmp_path, spec_dict=None, name="droid_franka"):
    spec_path = DROID_SPEC if spec_dict is None else write_spec(tmp_path, spec_dict, name)
    robot_out = tmp_path / "configs/robot_configs" / f"{name}.yaml"
    train_out = tmp_path / "configs/vla" / f"{name}.yaml"
    result = generate_configs(v3_data, spec_path, robot_out, train_out)
    return result, robot_out, train_out


def write_spec(tmp_path, spec_dict, name="spec"):
    spec_path = tmp_path / f"{name}_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")
    return spec_path


def droid_spec_dict():
    return yaml.safe_load(DROID_SPEC.read_text(encoding="utf-8"))


def make_r1pro_dataset(tmp_path):
    """Generation and validation only read meta/info.json; no parquet rows needed."""
    dataset = tmp_path / "r1pro_v3"
    (dataset / "meta").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [16]},
        "action": {"dtype": "float32", "shape": [16]},
        "observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]},
    }
    (dataset / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v3.0", "fps": 15, "features": features}), encoding="utf-8"
    )
    return dataset


def test_generated_pair_passes_validate_lingbot(v3_data, tmp_path):
    result, robot_out, train_out = generate(v3_data, tmp_path)
    assert result["data_name"] == "droid_franka"
    assert validate_mapping(v3_data, robot_out, train_out)["data_name"] == "droid_franka"
    robot = yaml.safe_load(robot_out.read_text(encoding="utf-8"))
    train = yaml.safe_load(train_out.read_text(encoding="utf-8"))
    assert [next(iter(item)) for item in robot["states"]] == [
        "observation.state.arm.position",
        "observation.state.effector.position",
    ]
    assert train["data"]["joints"] == [{"arm.position": 14}, {"effector.position": 2}]
    assert train["data"]["train_path"] == str(Path(v3_data).resolve())
    assert train["data"]["norm_stats_file"] == "assets/norm_stats/droid_franka.json"
    assert train["train"]["output_dir"] == "output/droid_franka"


def test_generated_configs_match_verified_handwritten_pair(v3_data, tmp_path):
    """The shipped spec must reproduce the hand-verified configs/robot_configs pair."""
    _, robot_out, train_out = generate(v3_data, tmp_path)
    hand_robot = yaml.safe_load(
        (PROJECT / "configs/robot_configs/droid_franka.yaml").read_text(encoding="utf-8")
    )
    assert yaml.safe_load(robot_out.read_text(encoding="utf-8")) == hand_robot
    hand_train = yaml.safe_load((PROJECT / "configs/vla/droid_franka.yaml").read_text(encoding="utf-8"))
    generated_train = yaml.safe_load(train_out.read_text(encoding="utf-8"))
    # Only the two path placeholders legitimately differ.
    for key in ["robot_config_root", "train_path"]:
        generated_train["data"].pop(key)
        hand_train["data"].pop(key)
    assert generated_train == hand_train


def test_capacity_defaults_to_mapped_width(v3_data, tmp_path):
    spec = droid_spec_dict()
    for joint in spec["joints"]:
        joint.pop("capacity")
    _, _, train_out = generate(v3_data, tmp_path, spec)
    train = yaml.safe_load(train_out.read_text(encoding="utf-8"))
    assert train["data"]["joints"] == [{"arm.position": 7}, {"effector.position": 1}]


def test_out_of_bounds_slice_rejected(v3_data, tmp_path):
    spec = droid_spec_dict()
    spec["joints"][0]["state"][0]["end"] = 9
    with pytest.raises(ValueError, match="out of bounds"):
        generate(v3_data, tmp_path, spec)


def test_missing_feature_rejected(v3_data, tmp_path):
    spec = droid_spec_dict()
    spec["joints"][0]["state"][0]["feature"] = "observation.left_arm"
    with pytest.raises(ValueError, match="Missing source feature"):
        generate(v3_data, tmp_path, spec)


def test_missing_camera_feature_rejected(v3_data, tmp_path):
    spec = droid_spec_dict()
    spec["cameras"][0]["feature"] = "observation.images.front"
    with pytest.raises(ValueError, match="Missing video feature"):
        generate(v3_data, tmp_path, spec)


def test_unmapped_gap_rejected_unless_disabled(v3_data, tmp_path):
    spec = droid_spec_dict()
    # Drop the gripper joint entirely -> observation.state[7:8] becomes unmapped.
    spec["joints"] = spec["joints"][:1]
    with pytest.raises(ValueError, match="Unmapped gap"):
        generate(v3_data, tmp_path, spec)
    spec["require_full_coverage"] = False
    result, _, _ = generate(v3_data, tmp_path, spec)
    assert result["data_name"] == "droid_franka"


def test_overlapping_slices_rejected_even_without_coverage(v3_data, tmp_path):
    spec = droid_spec_dict()
    spec["require_full_coverage"] = False
    spec["joints"][1]["state"][0]["start"] = 0
    spec["joints"][1]["state"][0]["end"] = 2
    spec["joints"][1]["action"][0]["start"] = 7
    spec["joints"][1]["action"][0]["end"] = 8
    with pytest.raises(ValueError, match="Overlapping"):
        generate(v3_data, tmp_path, spec)


def test_existing_outputs_never_overwritten(v3_data, tmp_path):
    robot_out = tmp_path / "configs/robot_configs/droid_franka.yaml"
    robot_out.parent.mkdir(parents=True)
    robot_out.write_text("keep me", encoding="utf-8")
    train_out = tmp_path / "configs/vla/droid_franka.yaml"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_configs(v3_data, DROID_SPEC, robot_out, train_out)
    assert robot_out.read_text(encoding="utf-8") == "keep me"
    assert not train_out.exists()


def test_managed_data_keys_rejected(v3_data, tmp_path):
    spec = droid_spec_dict()
    spec["data"]["data_name"] = "other"
    with pytest.raises(ValueError, match="cannot be overridden"):
        generate(v3_data, tmp_path, spec)


def test_robot_filename_must_match_spec_name(v3_data, tmp_path):
    with pytest.raises(ValueError, match="must match the spec name"):
        generate_configs(
            v3_data,
            DROID_SPEC,
            tmp_path / "configs/robot_configs/other.yaml",
            tmp_path / "configs/vla/droid_franka.yaml",
        )


def test_non_v3_dataset_rejected(tmp_path):
    dataset = tmp_path / "v21"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{"codebase_version": "v2.1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="requires LeRobot v3.0"):
        generate_configs(dataset, DROID_SPEC, tmp_path / "r.yaml", tmp_path / "t.yaml")


def test_capacity_exceeding_model_dims_rejected(v3_data, tmp_path):
    spec = droid_spec_dict()
    spec["joints"][0]["capacity"] = 70
    spec["joints"][1]["capacity"] = 70
    with pytest.raises(ValueError, match="exceeds"):
        generate(v3_data, tmp_path, spec)


def test_prefixed_joint_name_rejected(v3_data, tmp_path):
    spec = droid_spec_dict()
    spec["joints"][0]["name"] = "observation.state.arm.position"
    with pytest.raises(ValueError, match="bare joint name"):
        generate(v3_data, tmp_path, spec)


def test_interleaved_dual_arm_with_delta_actions(tmp_path):
    dataset = make_r1pro_dataset(tmp_path)
    robot_out = tmp_path / "configs/robot_configs/r1pro.yaml"
    train_out = tmp_path / "configs/vla/r1pro.yaml"
    result = generate_configs(dataset, R1PRO_SPEC, robot_out, train_out)
    assert result["mapped_dimensions"]["action.arm.position"] == 14
    robot = yaml.safe_load(robot_out.read_text(encoding="utf-8"))
    arm = robot["actions"][0]["action.arm.position"]
    assert arm["subtract_state"] is True
    assert [list(item)[0] for item in arm["origin_keys"]] == ["action", "action"]
    assert [next(iter(item.values())) for item in arm["origin_keys"]] == [
        {"start": 0, "end": 7},
        {"start": 8, "end": 15},
    ]
    assert robot["actions"][1]["action.effector.position"]["subtract_state"] is False
    train = yaml.safe_load(train_out.read_text(encoding="utf-8"))
    assert train["data"]["joints"] == [{"arm.position": 14}, {"effector.position": 2}]
    assert train["data"]["num_workers"] == 8
    assert train["train"]["micro_batch_size"] == 32
    assert validate_mapping(dataset, robot_out, train_out)["cameras"] == ["camera_top"]


def test_cli_generates_config_pair(v3_data, tmp_path):
    from typer.testing import CliRunner

    runner = CliRunner()
    robot_out = tmp_path / "configs/robot_configs/droid_franka.yaml"
    train_out = tmp_path / "configs/vla/droid_franka.yaml"
    result = runner.invoke(
        app,
        [
            "generate-lingbot-config",
            str(v3_data),
            "--spec",
            str(DROID_SPEC),
            "--robot-config",
            str(robot_out),
            "--train-config",
            str(train_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert robot_out.is_file() and train_out.is_file()
    assert "droid_franka" in result.output


def test_cli_failure_exits_nonzero(v3_data, tmp_path):
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["generate-lingbot-config", str(v3_data), "--spec", str(tmp_path / "missing.yaml")],
    )
    assert result.exit_code != 0
