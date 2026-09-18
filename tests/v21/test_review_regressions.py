"""Review regressions and a real video-free end-to-end pipeline run."""
import json
from unittest.mock import patch

import conftest
import pandas as pd
import pytest
from test_numeric_stages import dataset, work

from lerobot_cleaner.v21.config import CleaningConfig, NumericSanityConfig, load_preset
from lerobot_cleaner.v21.pipeline import Pipeline
from lerobot_cleaner.v21.rules.checks.trajectory.outlier import OutlierRule


@pytest.fixture
def no_video_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(conftest, "_has_ffmpeg", lambda: True)
    monkeypatch.setattr(conftest, "_write_video", lambda *args: None)
    root = conftest.synth_dataset.__wrapped__(tmp_path)
    info = json.loads((root / "meta/info.json").read_text())
    info["features"] = {k: v for k, v in info["features"].items() if v["dtype"] != "video"}
    info["total_videos"] = 0
    (root / "meta/info.json").write_text(json.dumps(info))
    modality = json.loads((root / "meta/modality.json").read_text())
    modality["video"] = {}
    (root / "meta/modality.json").write_text(json.dumps(modality))
    return root


def test_builtin_preset_loads_after_version_split():
    assert load_preset("galaxea_r1pro_single_arm")


def test_missing_bounds_are_not_a_passing_quality_check():
    result = OutlierRule(NumericSanityConfig(), dataset()).run(work([1, 2]))
    assert not result.passed and result.severity == "warning"
    assert result.metrics["state"]["evaluated"] is False


@pytest.mark.parametrize("resume", [False, True])
def test_existing_output_is_not_overwritten(no_video_dataset, tmp_path, resume):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep")
    cfg = CleaningConfig(input=no_video_dataset, output=out, resume=resume)
    with pytest.raises(ValueError, match="new output directory"):
        Pipeline(cfg).run()
    assert sentinel.read_text() == "keep"


def test_pipeline_finalizers_and_quality_reports_end_to_end(no_video_dataset, tmp_path):
    out = tmp_path / "clean"
    cfg = CleaningConfig(input=no_video_dataset, output=out, num_workers=2,
        rules={"numeric_sanity": {"enabled": True, "on_nan": "drop_frame", "outlier_mode": "off"},
               "episode_length_filter": {"enabled": True, "min_frames": 30}})
    # Avoid optional plotting cost; workers, parquet IO, finalizers, and reports are real.
    with patch("lerobot_cleaner.v21.report.CleaningReport._write_figures"):
        report = Pipeline(cfg).run()
    assert report.kept_episodes == 2
    info = json.loads((out / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (out / "meta/episodes.jsonl").read_text().splitlines()]
    assert [ep["episode_index"] for ep in episodes] == [0, 1]
    frames = [pd.read_parquet(out / info["data_path"].format(episode_chunk=0, episode_index=i)) for i in range(2)]
    assert pd.concat(frames)["index"].tolist() == list(range(info["total_frames"]))
    quality = [json.loads(line) for line in (out / "cleaning_report/episode_quality.jsonl").read_text().splitlines()]
    assert [rec["episode_index"] for rec in quality] == [0, 1, 2]
    assert quality[1]["dropped"]
    assert quality[2]["transforms"]["numeric_repair"]["changed"]
    assert (out / "meta/stats.json").is_file()
    assert (out / "meta/episodes_stats.jsonl").is_file()
