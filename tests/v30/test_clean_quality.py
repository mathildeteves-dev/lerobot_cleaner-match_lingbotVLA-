"""Clean reports retain quality of raw input and serialized output independently."""
import json

import numpy as np
import pytest

from lerobot_cleaner.v30 import v3_streaming as streaming
from lerobot_cleaner.v30.v3 import audit_v3, clean_v3
from tests.v30.test_audit_quality import fingerprints
from tests.v30.test_v3 import alias_config, mutate_actions


def config(engine, enabled=True):
    return alias_config(
        engine=engine, progress=False, disk_reserve_gb=0,
        nonfinite="interpolate", bounds={"action": (0, 0.2)},
        quality={
            "enabled": enabled,
            "groups": [
                {"name": "state_arm", "source": "state", "columns": [0], "velocity": 1},
                {"name": "action_arm", "source": "action", "columns": [0], "velocity": 1},
            ],
            "joint_static_ratio": {"state_epsilon": 0.01, "action_epsilon": 0.01},
        },
    )


@pytest.mark.parametrize("engine", ["memory", "streaming"])
@pytest.mark.parametrize("enabled", [True, False])
def test_clean_quality_matches_raw_and_written_audits(v3_data, tmp_path, engine, enabled):
    mutate_actions(v3_data, lambda a: a.__setitem__((1, 0), np.nan))
    before = fingerprints(v3_data)
    cfg = config(engine, enabled)
    expected_input = audit_v3(v3_data, cfg)["trajectory_quality"]
    output = tmp_path / "clean"
    report = clean_v3(v3_data, output, cfg)
    expected_output = audit_v3(output, cfg)["trajectory_quality"]
    persisted = json.loads((output / "cleaning_report/report.json").read_text())
    for value in (report, persisted):
        assert "trajectory_quality" not in value
        assert value["trajectory_quality_input"] == expected_input
        assert value["trajectory_quality_output"] == expected_output
    if enabled:
        assert len(expected_input) == len(expected_output) == 2
        assert not expected_input[0]["checks"]["finite"]["passed"]
        assert expected_output[0]["checks"]["finite"]["passed"]
        assert expected_input != expected_output
    else:
        assert expected_input == expected_output == []
    assert fingerprints(v3_data) == before


def test_clean_quality_engine_parity(v3_data, tmp_path):
    reports = [clean_v3(v3_data, tmp_path / engine, config(engine))
               for engine in ("memory", "streaming")]
    for key in ("trajectory_quality_input", "trajectory_quality_output"):
        assert reports[0][key] == reports[1][key]


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_resume_preserves_or_recovers_raw_quality(v3_data, tmp_path, monkeypatch, legacy_checkpoint):
    mutate_actions(v3_data, lambda a: a.__setitem__((1, 0), np.nan))
    cfg = config("streaming")
    expected_input = audit_v3(v3_data, cfg)["trajectory_quality"]
    output = tmp_path / "clean"
    original_copy = streaming.copy_verified

    def interrupt(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(streaming, "copy_verified", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        clean_v3(v3_data, output, cfg)
    checkpoint = tmp_path / "clean.partial/scan.json"
    if legacy_checkpoint:
        saved = json.loads(checkpoint.read_text())
        del saved["report"]["trajectory_quality_input"]
        saved["report"]["trajectory_quality"] = []
        checkpoint.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(streaming, "copy_verified", original_copy)
    report = clean_v3(v3_data, output, cfg, resume=True)
    assert "trajectory_quality" not in report
    assert report["trajectory_quality_input"] == expected_input
    assert report["trajectory_quality_output"] == audit_v3(output, cfg)["trajectory_quality"]
    assert report["trajectory_quality_input"] != report["trajectory_quality_output"]
    saved = json.loads(checkpoint.read_text())
    assert saved["report"]["trajectory_quality_input"] == expected_input

@pytest.mark.parametrize("engine", ["memory", "streaming"])
def test_unchanged_clean_has_equal_nonempty_quality(v3_data, tmp_path, engine):
    cfg = config(engine).model_copy(update={"bounds": {}})
    report = clean_v3(v3_data, tmp_path / "clean", cfg)
    assert report["changed_values"] == {}
    assert len(report["trajectory_quality_input"]) == 2
    assert report["trajectory_quality_input"] == report["trajectory_quality_output"]
