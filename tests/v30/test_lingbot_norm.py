"""Regression coverage for LingBot mapping and normalization launch failures."""

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from test_v3 import PROJECT

from lerobot_cleaner.v30.lingbot.config import validate_mapping
from scripts import run_lingbot_norm as norm


@pytest.fixture
def setup_norm(v3_data, tmp_path):  # noqa: F811
    lingbot = tmp_path / "lingbot with spaces"
    (lingbot / "scripts").mkdir(parents=True)
    (lingbot / "train.sh").touch()
    (lingbot / "scripts/compute_norm.py").touch()
    robot = PROJECT / "configs/robot_configs/droid_franka.yaml"
    train = PROJECT / "configs/vla/droid_franka.yaml"
    output = tmp_path / "output with spaces" / "norm.json"
    return lingbot, v3_data, robot, train, output


def valid_stats():
    return {
        "count": 6,
        "norm_stats": {
            key: {
                name: [0.0] * width
                for name in ["mean", "std", "min", "max", "q01", "q02", "q98", "q99"]
            }
            for key, width in [
                ("action.arm.position", 7),
                ("action.effector.position", 1),
                ("observation.state.arm.position", 7),
                ("observation.state.effector.position", 1),
            ]
        },
    }


@pytest.mark.parametrize("joint", ["effector.position", "end.position"])
def test_forbidden_delta_rejected(setup_norm, tmp_path, joint):
    _, dataset, robot_path, train_path, _ = setup_norm
    robot = yaml.safe_load(robot_path.read_text(encoding="utf-8"))
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    for section, prefix in [("states", "observation.state."), ("actions", "action.")]:
        item = robot[section][1]
        value = item.pop(prefix + "effector.position")
        item[prefix + joint] = value
        if section == "actions":
            value["subtract_state"] = True
    train["data"]["joints"][1] = {joint: 2}
    robot_path = tmp_path / "droid_franka.yaml"
    train_path = tmp_path / "train.yaml"
    robot_path.write_text(yaml.safe_dump(robot), encoding="utf-8")
    train_path.write_text(yaml.safe_dump(train), encoding="utf-8")
    with pytest.raises(ValueError, match="LingBot forbids subtract_state"):
        validate_mapping(dataset, robot_path, train_path)


def test_arm_delta_and_chunk_statistics_accepted(setup_norm, tmp_path):
    _, dataset, robot_path, train_path, output = setup_norm
    robot = yaml.safe_load(robot_path.read_text(encoding="utf-8"))
    robot["actions"][0]["action.arm.position"]["subtract_state"] = True
    robot_path = tmp_path / "droid_franka.yaml"
    robot_path.write_text(yaml.safe_dump(robot), encoding="utf-8")
    result = valid_stats()
    result["norm_stats"]["action.arm.position"] = {
        key: [copy.deepcopy(value) for _ in range(50)]
        for key, value in result["norm_stats"]["action.arm.position"].items()
    }
    output.parent.mkdir()
    output.write_text(json.dumps(result), encoding="utf-8")
    norm.validate_norm_output(output, dataset, robot_path, train_path)


@pytest.mark.parametrize("device", ["0,1", "", " ", "-1", "all", "0\n1"])
def test_invalid_devices_never_launch(setup_norm, monkeypatch, device):
    monkeypatch.setattr(norm.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(argparse.ArgumentTypeError, match="exactly one GPU"):
        norm.run_normalization(*setup_norm, cuda_devices=device)
    assert not setup_norm[-1].exists()


@pytest.mark.parametrize("device", ["0", " 2 ", "GPU-abcd-123", "MIG-GPU-abcd/1/0"])
def test_single_device_identifiers(device):
    assert norm.single_cuda_device(device) == device.strip()


def test_success_is_validated_and_published_with_one_process(setup_norm, monkeypatch):
    output = setup_norm[-1]
    monkeypatch.setenv("MASTER_ADDR", "remote-node")
    for key in ["NNODES", "NODE_RANK", "NPROC_PER_NODE"]:
        monkeypatch.setenv(key, "8")

    def run(command, **kwargs):
        env = kwargs["env"]
        assert [env[k] for k in ["NNODES", "NODE_RANK", "NPROC_PER_NODE"]] == ["1", "0", "1"]
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        assert env["MASTER_ADDR"] == "127.0.0.1"
        assert kwargs["cwd"] == setup_norm[0].resolve()
        assert kwargs["check"]
        assert command[:3] == ["bash", "-o", "pipefail"]
        assert not output.exists()
        Path(command[-1]).write_text(json.dumps(valid_stats()), encoding="utf-8")

    monkeypatch.setattr(norm.subprocess, "run", run)
    norm.run_normalization(*setup_norm, cuda_devices="2")
    assert json.loads(output.read_text()) == valid_stats()
    assert not list(output.parent.glob(".lingbot-norm-*"))


@pytest.mark.parametrize(
    "failure",
    [
        "process",
        "missing",
        "json",
        "keys",
        "width",
        "nan",
        "inf",
        "negative_std",
        "count",
        "bool_count",
        "missing_field",
        "string",
        "bounds",
        "quantiles",
    ],
)
def test_failed_or_invalid_statistics_are_not_published(setup_norm, monkeypatch, failure):
    def run(command, **kwargs):
        result = valid_stats()
        fields = result["norm_stats"]["action.effector.position"]
        if failure == "missing":
            return
        if failure == "json":
            Path(command[-1]).write_text('{"norm_stats":', encoding="utf-8")
            return
        if failure == "keys":
            result["norm_stats"].pop("action.arm.position")
        elif failure == "width":
            fields["mean"] = [0.0, 0.0]
        elif failure in {"nan", "inf"}:
            fields["mean"] = [float(failure)]
        elif failure == "negative_std":
            fields["std"] = [-1.0]
        elif failure == "count":
            result["count"] = 5
        elif failure == "bool_count":
            result["count"] = True
        elif failure == "missing_field":
            fields.pop("q99")
        elif failure == "string":
            fields["mean"] = ["0"]
        elif failure == "bounds":
            fields["min"] = [1.0]
        elif failure == "quantiles":
            fields["q01"] = [1.0]
        Path(command[-1]).write_text(json.dumps(result), encoding="utf-8")
        if failure == "process":
            raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(norm.subprocess, "run", run)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        norm.run_normalization(*setup_norm)
    assert not setup_norm[-1].exists()
    assert not list(setup_norm[-1].parent.glob(".lingbot-norm-*"))


@pytest.mark.parametrize("appears_during_run", [False, True])
def test_existing_output_never_overwritten(setup_norm, monkeypatch, appears_during_run):
    output = setup_norm[-1]
    output.parent.mkdir()
    if not appears_during_run:
        output.write_text("keep me", encoding="utf-8")

    def run(command, **kwargs):
        assert appears_during_run, "existing output should prevent launch"
        Path(command[-1]).write_text(json.dumps(valid_stats()), encoding="utf-8")
        output.write_text("keep me", encoding="utf-8")

    monkeypatch.setattr(norm.subprocess, "run", run)
    with pytest.raises(FileExistsError):
        norm.run_normalization(*setup_norm)
    assert output.read_text() == "keep me"


def test_pipefail_propagates_failure_through_tee(setup_norm):
    bash = shutil.which("bash")
    if not bash:
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(candidate) if candidate.is_file() else None
    if not bash:
        pytest.skip("Bash unavailable; pipeline integration test needs Bash")
    # Git Bash does not necessarily inherit its Unix utilities on Windows PATH.
    # A login shell initializes that PATH, which is required to exercise tee.
    root, dataset, robot, train, output = setup_norm
    # Reproduce upstream's final pipeline, without requiring torch, CUDA or LingBot imports.
    (root / "train.sh").write_text("#!/bin/bash\n(exit 7) 2>&1 | tee log.txt\n", encoding="utf-8")
    command = norm.build_command(root, dataset, robot, train, output)
    command[0] = bash
    command.insert(1, "-l")
    with pytest.raises(subprocess.CalledProcessError) as exc:
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    assert exc.value.returncode == 7
