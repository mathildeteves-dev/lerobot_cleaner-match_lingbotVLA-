"""Robust group calibration and read-only -> config -> clean integration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from lerobot_cleaner.cli import app
from lerobot_cleaner.core.quality.calibration import robust_threshold
from lerobot_cleaner.v30.calibration import CalibrationConfig, calibrate_v3
from lerobot_cleaner.v30.v3 import V3Config
from tests.v30.test_audit_quality import fingerprints


def test_quantile_and_mad_conservative_max():
    values = np.array([1, 2, 3, 4, 5.])
    result = robust_threshold(values, min_samples=2)
    assert result["median"] == 3
    assert result["mad"] == 1
    assert result["threshold"] == pytest.approx(3 + 8 * 1.4826)
    tail = robust_threshold([0, 0, 0, 100], min_samples=2)
    assert tail["mad_threshold"] == 0
    assert tail["threshold"] == np.quantile([0, 0, 0, 100], .995)
    np.testing.assert_array_equal(values, [1, 2, 3, 4, 5])


def test_invalid_and_constant_populations():
    report = robust_threshold([0, 0, np.nan, np.inf, -1], min_samples=2)
    assert report["count"] == 2 and report["excluded"] == 3
    assert report["floor_applied"] and report["threshold"] == 1e-12
    insufficient = robust_threshold([np.nan, 4], min_samples=2)
    assert insufficient["status"] == "insufficient_samples"
    assert insufficient["threshold"] is None


@pytest.mark.parametrize("kwargs", [dict(quantile=1), dict(quantile=0), dict(mad_k=-1),
                                     dict(min_samples=1), dict(positive_floor=0)])
def test_invalid_estimator_parameters(kwargs):
    with pytest.raises(ValueError):
        robust_threshold([1, 2], **kwargs)


@pytest.fixture
def calibration_data(v3_data):
    # Four samples/episode allow jerk; each group has distinct scale.
    path = v3_data / "data/chunk-000/file-000.parquet"
    original = pd.read_parquet(path)
    frames = []
    for ep in range(2):
        part = original[original.episode_index == ep].iloc[[0, 1, 2, 2]].copy()
        t = np.arange(4, dtype=float)
        action = np.repeat(((ep + 1) * t**3)[:, None], 8, axis=1).astype(np.float32)
        action[:, 7] *= 10
        state = action * 2
        part["action"] = list(action)
        part["action.joint_position"] = list(action[:, :7])
        part["action.gripper_position"] = action[:, 7]
        part["observation.state"] = list(state)
        part["index"] = np.arange(4) + ep * 4
        part["frame_index"] = np.arange(4)
        part["timestamp"] = (t / 15).astype(np.float32)
        frames.append(part)
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    info_path = v3_data / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 8
    info_path.write_text(json.dumps(info))
    ep_path = v3_data / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(ep_path)
    episodes["length"] = [4, 4]
    episodes["dataset_from_index"] = [0, 4]
    episodes["dataset_to_index"] = [4, 8]
    for column in episodes:
        if column.endswith("/from_timestamp"):
            episodes[column] = [0., 4/15]
        if column.endswith("/to_timestamp"):
            episodes[column] = [4/15, 8/15]
    episodes.to_parquet(ep_path, index=False)
    return v3_data


def config(engine):
    return V3Config(engine=engine, disk_reserve_gb=0, quality={
        "groups": [{"name": f"{source}_{name}", "source": source, "columns": columns,
                    "velocity": .01, "jerk": .01}
                   for source in ["state", "action"]
                   for name, columns in [("arm", list(range(7))), ("gripper", [7])]],
        "joint_static_ratio": {"state_epsilon": .01, "action_epsilon": .02},
    })


@pytest.mark.parametrize("engine", ["memory", "streaming"])
def test_calibration_config_roundtrip_and_optional_clean(calibration_data, tmp_path, engine):
    before = fingerprints(calibration_data)
    cfg = config(engine)
    report = calibrate_v3(calibration_data, tmp_path / "calibration", cfg,
                          CalibrationConfig(min_samples=2), clean_output=tmp_path / "clean")
    assert report["status"] == "ready"
    assert report["cleaning"]["status"] == "passed"
    assert report["cleaning"]["frames"] == 8
    assert fingerprints(calibration_data) == before
    import hashlib
    assert report["threshold_config_sha256"] == hashlib.sha256(Path(report["threshold_config"]).read_bytes()).hexdigest()
    generated = V3Config.from_yaml(Path(report["threshold_config"]))
    assert generated.quality.joint_static_ratio == cfg.quality.joint_static_ratio
    assert generated.bounds == cfg.bounds
    thresholds = {g.name: g.velocity for g in generated.quality.groups}
    assert thresholds["state_arm"] == pytest.approx(thresholds["action_arm"] * 2)
    assert thresholds["state_gripper"] == pytest.approx(thresholds["state_arm"] * 10)
    assert cfg.quality.groups[0].velocity == .01  # original config is not mutated
    cleaned = json.loads((tmp_path / "clean/cleaning_report/report.json").read_text())
    check = cleaned["trajectory_quality_input"][0]["groups"]["state_arm"]["checks"]["velocity"]
    assert check["metrics"]["threshold"] == thresholds["state_arm"]
    assert cleaned["changed_values"] == {} and cleaned["rows_removed"] == 0


def test_calibration_only_and_engine_parity(calibration_data, tmp_path):
    before = fingerprints(calibration_data)
    reports = [calibrate_v3(calibration_data, tmp_path / engine, config(engine), CalibrationConfig(min_samples=2))
               for engine in ["memory", "streaming"]]
    assert reports[0]["groups"] == reports[1]["groups"]
    assert reports[0]["cleaning"]["status"] == "not_requested"
    assert fingerprints(calibration_data) == before


def test_short_episodes_report_exclusions_and_block_clean(v3_data, tmp_path):
    before = fingerprints(v3_data)
    report = calibrate_v3(v3_data, tmp_path / "cal", config("streaming"),
                          CalibrationConfig(min_samples=2), clean_output=tmp_path / "clean")
    assert report["status"] == "insufficient_calibration"
    assert report["groups"]["state_arm"]["metrics"]["jerk"]["excluded_episodes"] == 2
    assert not (tmp_path / "cal/thresholds.yaml").exists()
    assert not (tmp_path / "clean").exists()
    assert (tmp_path / "cal/calibration_report.json").is_file()
    assert fingerprints(v3_data) == before


def test_groups_required_and_input_output_paths_protected(v3_data, tmp_path):
    with pytest.raises(ValueError, match="groups"):
        calibrate_v3(v3_data, tmp_path / "cal", V3Config())
    with pytest.raises(ValueError, match="outside"):
        calibrate_v3(v3_data, v3_data / "cal", config("memory"))
    with pytest.raises(ValueError, match="separate"):
        calibrate_v3(v3_data, tmp_path / "cal", config("memory"), clean_output=tmp_path / "cal/clean")


def test_cli_uses_generated_config(calibration_data, tmp_path):
    import yaml
    cfg = tmp_path / "input.yaml"
    cfg.write_text(yaml.safe_dump(config("streaming").model_dump(mode="json")))
    result = CliRunner().invoke(app, ["calibrate-v3", str(calibration_data), "--config", str(cfg),
        "--output", str(tmp_path / "cal"), "--min-samples", "2", "--clean-output", str(tmp_path / "clean")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "clean/cleaning_report/report.json").is_file()


def test_nonfinite_is_excluded_only_from_affected_group(calibration_data, tmp_path):
    path = calibration_data / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(path)
    state = data.at[0, "observation.state"].copy()
    state[7] = np.nan
    data.at[0, "observation.state"] = state
    data.to_parquet(path, index=False)
    before = fingerprints(calibration_data)
    cfg = config("streaming").model_copy(update={"nonfinite": "interpolate"})
    report = calibrate_v3(calibration_data, tmp_path / "cal", cfg, CalibrationConfig(min_samples=2))
    assert report["groups"]["state_gripper"]["metrics"]["velocity"]["count"] == 1
    assert report["groups"]["state_arm"]["metrics"]["velocity"]["count"] == 2
    assert report["groups"]["action_gripper"]["metrics"]["velocity"]["count"] == 2
    assert report["status"] == "insufficient_calibration"
    assert fingerprints(calibration_data) == before  # interpolate was NOT executed


def test_source_change_blocks_publishing(calibration_data, tmp_path, monkeypatch):
    import lerobot_cleaner.v30.calibration as module
    original = module.audit_v3

    def changed(root, cfg):
        result = original(root, cfg)
        with (root / "meta/stats.json").open("a") as handle:
            handle.write(" ")
        return result
    monkeypatch.setattr(module, "audit_v3", changed)
    with pytest.raises(ValueError, match="Source changed"):
        calibrate_v3(calibration_data, tmp_path / "cal", config("memory"), CalibrationConfig(min_samples=2))
    assert not (tmp_path / "cal").exists()


def test_cleaning_failure_preserves_calibration_report(calibration_data, tmp_path, monkeypatch):
    import lerobot_cleaner.v30.calibration as module

    def fail(*args):
        raise RuntimeError("synthetic cleaning failure")
    monkeypatch.setattr(module, "clean_v3", fail)
    report = calibrate_v3(calibration_data, tmp_path / "cal", config("memory"),
                          CalibrationConfig(min_samples=2), clean_output=tmp_path / "clean")
    assert report["status"] == "cleaning_failed"
    assert report["cleaning"]["status"] == "failed"
    assert (tmp_path / "cal/thresholds.yaml").is_file()
    saved = json.loads((tmp_path / "cal/calibration_report.json").read_text(encoding="utf-8"))
    assert saved["cleaning"]["error"] == "synthetic cleaning failure"


def test_cli_insufficient_calibration_is_nonzero(v3_data, tmp_path):
    import yaml
    cfg = tmp_path / "input.yaml"
    cfg.write_text(yaml.safe_dump(config("memory").model_dump(mode="json")))
    result = CliRunner().invoke(app, ["calibrate-v3", str(v3_data), "--config", str(cfg),
        "--output", str(tmp_path / "cal"), "--min-samples", "2"])
    assert result.exit_code == 2, result.output
    assert (tmp_path / "cal/calibration_report.json").is_file()
    assert not (tmp_path / "cal/thresholds.yaml").exists()


def test_lingbot_adapter_survives_generated_config_roundtrip(calibration_data, tmp_path):
    robot = Path(__file__).resolve().parents[2] / "configs/robot_configs/droid_franka.yaml"
    cfg = config("streaming").model_copy(update={"robot_config": robot})
    report = calibrate_v3(calibration_data, tmp_path / "cal", cfg,
                          CalibrationConfig(min_samples=2), clean_output=tmp_path / "clean")
    assert report["status"] == "ready"
    generated = V3Config.from_yaml(tmp_path / "cal/thresholds.yaml")
    assert generated.robot_config == robot
    cleaned = json.loads((tmp_path / "clean/cleaning_report/report.json").read_text(encoding="utf-8"))
    assert cleaned["adapter"]["type"] == "LingBotAdapter"
    assert not (tmp_path / "clean/meta/modality.json").exists()
