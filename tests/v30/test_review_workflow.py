import json
import subprocess
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_libero

from lerobot_cleaner.v30.episode_review import DatasetReview
from lerobot_cleaner.v30.libero import validate_libero
from lerobot_cleaner.v30.review_profile import ReviewProfile, load_profile
from lerobot_cleaner.v30.review_report import report_status
from lerobot_cleaner.v30.training_readiness import check_readiness
from lerobot_cleaner.v30.v3 import V3Config
from lerobot_cleaner.v30.v3_streaming import audit_streaming, clean_streaming, verify_videos
from lerobot_cleaner.v30.video_review import VideoReview

libero_data = test_libero.libero_data
PROJECT = Path(__file__).parents[2]


def config():
    return V3Config(engine="streaming", disk_reserve_gb=0, batch_rows=2, metadata_batch_rows=1)


def run_cli(dataset, output, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(PROJECT / "scripts/run_libero_clean.py"),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--quiet",
            *extra,
        ],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_audit_reports_nan_and_returns_failure(libero_data, tmp_path):
    path = libero_data / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path)
    actions = np.stack(frame.action)
    actions[0, 0] = np.nan
    frame["action"] = list(actions)
    frame.to_parquet(path, index=False)
    output = tmp_path / "audit"
    run = run_cli(libero_data, output, "--audit-only", "--quick")
    assert run.returncode == 2, run.stdout + run.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["review_summary"]["nonfinite"] == {"action": 1}
    assert '"status": "failed"' in run.stdout
    assert '"action": 1' in run.stdout
    assert not (output / "COMPLETE.json").exists()


def test_clean_report_failure_does_not_publish_and_can_resume(libero_data, tmp_path):
    output = tmp_path / "clean"

    def broken(stage, report):
        raise OSError("report disk error")

    with pytest.raises(OSError, match="report disk"):
        clean_streaming(libero_data, output, config(), finalize=broken)
    assert not output.exists()
    assert not (tmp_path / "clean.partial/dataset/cleaning_report/COMPLETE.json").exists()
    clean_streaming(libero_data, output, config(), resume=True)
    assert (
        json.loads((output / "cleaning_report/COMPLETE.json").read_text())["status"] == "complete"
    )


def test_cli_emits_automatic_bundle(libero_data, tmp_path):
    output = tmp_path / "clean"
    run = run_cli(libero_data, output, "--quick")
    assert run.returncode == 0, run.stdout + run.stderr
    folder = output / "cleaning_report"
    for filename in [
        "COMPLETE.json",
        "analysis_zh.md",
        "review.html",
        "profile.used.yaml",
        "post_validation.json",
        "training_readiness.json",
        "episode_quality.json",
    ]:
        assert (folder / filename).is_file(), filename
    assert '"status": "warning"' in run.stdout  # Quick must not imply full video validation.
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert ".partial" not in report["output_review"]["input"]


def test_alias_float_tolerance_does_not_change_data(libero_data, tmp_path):
    path = libero_data / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path)
    ee = np.stack(frame["observation.states.ee_state"])
    ee[0, 0] += 1e-8
    frame["observation.states.ee_state"] = list(ee)
    frame.to_parquet(path, index=False)
    validate_libero(libero_data)
    output = tmp_path / "clean"
    clean_streaming(libero_data, output, config(), observer_factory=DatasetReview)
    pd.testing.assert_frame_equal(
        frame, pd.read_parquet(output / "data/chunk-000/file-000.parquet")
    )


def test_profile_accepts_explicit_camera_rename(libero_data):
    info_path = libero_data / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.front"] = info["features"].pop("observation.images.image")
    info_path.write_text(json.dumps(info))
    profile = load_profile().model_copy(
        update={"cameras": ["observation.images.front", "observation.images.wrist_image"]}
    )
    DatasetReview(libero_data, profile)
    with pytest.raises(ValueError, match="cameras"):
        DatasetReview(libero_data)


def test_quality_marks_static_and_duplicate_without_deleting(libero_data):
    path = libero_data / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path)
    for key in [
        "observation.state",
        "observation.states.ee_state",
        "observation.states.gripper_state",
        "action",
    ]:
        frame[key] = [np.zeros(len(frame[key].iloc[0]), dtype=np.float32)] * len(frame)
    frame.to_parquet(path, index=False)
    profile = load_profile()
    profile.quality.static_seconds = 0.05
    report = audit_streaming(libero_data, config(), observer=DatasetReview(libero_data, profile))
    rows = report["dataset_review"]["episode_quality"]
    assert "long_static_state" in rows[0]["flags"]
    assert rows[1]["duplicate_numeric_trajectory_of"] == 0
    assert report["frames"] == 6


def test_binary_gripper_switch_not_treated_as_motion_jump(libero_data):
    path = libero_data / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path)
    a = np.zeros((len(frame), 7), dtype=np.float32)
    a[:, -1] = [0, 1, 0, 1, 0, 1]
    frame["action"] = list(a)
    frame.to_parquet(path, index=False)
    report = validate_libero(libero_data)
    assert all(row["action_jump_count"] == 0 for row in report["episode_quality"])


def test_video_decode_checkpoint_reused_and_invalidated(v3_data, tmp_path, monkeypatch):
    calls = []

    class Container:
        streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=Fraction(1, 15))])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            for i in range(6):
                yield types.SimpleNamespace(pts=i, height=180, width=320)

    def opened(path):
        calls.append(path)
        return Container()

    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=opened))
    info = json.loads((v3_data / "meta/info.json").read_text())
    cfg = config()
    report = audit_streaming(v3_data, cfg)
    cache = tmp_path / "decode.json"
    verify_videos(v3_data, info, report["videos"], cfg, checkpoint=cache)
    first = len(calls)
    verify_videos(v3_data, info, report["videos"], cfg, checkpoint=cache)
    assert len(calls) == first
    assert all(item["decode_reused"] for item in report["videos"].values())
    relative = next(iter(report["videos"]))
    (v3_data / relative).write_bytes(b"changed-video-content")
    verify_videos(v3_data, info, report["videos"], cfg, checkpoint=cache)
    assert len(calls) == first + 1
    assert report["videos"][relative]["decoded_frames"] == 6


def test_visual_sampling_marks_black_static_and_saves_preview(tmp_path):
    from PIL import Image

    options = load_profile().quality.visual.model_dump()
    options["preview_episodes"] = [0]
    frame = types.SimpleNamespace(
        to_ndarray=lambda format: np.zeros((64, 64), dtype=np.uint8),
        to_image=lambda: Image.new("RGB", (64, 64)),
    )
    review = VideoReview(
        tmp_path, "x.mp4", {"episode_indices": [0], "intervals": [[0, 5]]}, {"fps": 1}, options
    )
    for i in range(5):
        review.sample(frame, 0, float(i), i + 1)
    row = review.result()["episodes"][0]
    assert "sampled_black_frames" in row["flags"]
    assert "sampled_unchanged_scene" in row["flags"]
    assert (tmp_path / row["preview"]).exists()


def test_no_fabricated_training_readiness(tmp_path):
    result = check_readiness(load_profile(), tmp_path)
    assert result["status"] == "blocked"
    assert not result["runtime_validated"]


def test_semantics_require_evidence():
    value = load_profile().model_dump()
    value["semantics"]["verified"] = True
    with pytest.raises(ValueError, match="evidence"):
        ReviewProfile.model_validate(value)


def test_report_status_nonfinite_takes_precedence():
    assert report_status({"numeric": {"action": {"nonfinite": 1}}})["status"] == "failed"


def test_decode_interruption_reuses_completed_files_and_persists_counts(v3_data, tmp_path, monkeypatch):
    calls = []
    fail_once = [True]

    class Container:
        streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=Fraction(1, 15))])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            for i in range(6):
                yield types.SimpleNamespace(pts=i, height=180, width=320)

    def opened(path):
        calls.append(path)
        if len(calls) == 2 and fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("interrupted video decode")
        return Container()

    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open=opened))
    output = tmp_path / "clean"
    cfg = config().model_copy(update={"verify_videos": True})
    with pytest.raises(RuntimeError, match="interrupted video"):
        clean_streaming(v3_data, output, cfg)
    assert not output.exists()
    assert len(json.loads((tmp_path / "clean.partial/video_decode.json").read_text())) == 1
    clean_streaming(v3_data, output, cfg, resume=True)
    report = json.loads((output / "cleaning_report/report.json").read_text())
    assert sum(v["decode_reused"] for v in report["videos"].values()) == 1
    assert all(v["decoded_frames"] == 6 for v in report["videos"].values())
    assert all(v["interval_frame_counts"] == [3, 3] for v in report["videos"].values())
    assert len(calls) == 4  # First complete file is not decoded twice.
