"""Per-call modification records and report propagation."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
from test_numeric_stages import BOUNDS, dataset, work

from lerobot_cleaner.v21.config import (
    GripperBinarizeConfig,
    NumericSanityConfig,
    StaticFrameTrimConfig,
    VideoRoiCropConfig,
)
from lerobot_cleaner.v21.pipeline import Pipeline
from lerobot_cleaner.v21.report import CleaningReport
from lerobot_cleaner.v21.rules import run_episode_stages
from lerobot_cleaner.v21.rules.base import TransformRule
from lerobot_cleaner.v21.rules.transforms.embodiment.gripper import GripperBinarizeRule
from lerobot_cleaner.v21.rules.transforms.motion.static_trim import StaticFrameTrimRule
from lerobot_cleaner.v21.rules.transforms.numeric.percentile_clip import PercentileClipRule
from lerobot_cleaner.v21.rules.transforms.numeric.repair import NumericRepairRule
from lerobot_cleaner.v21.rules.transforms.vision.roi_crop import VideoRoiCropRule
from lerobot_cleaner.v21.types import CheckResult, TransformResult


def test_static_trim_records_local_positions_even_with_source_gaps():
    episode = work([0, 0, 0, 1, 2, 2, 2])
    episode.keep_indices = [0, 2, 4, 6, 8, 10, 12]
    rule = StaticFrameTrimRule(StaticFrameTrimConfig(pos_threshold=.01, rot_threshold_deg=None), dataset())
    result = rule.apply(episode)
    assert result.episode is episode and result.changed
    assert result.metrics == {"frames_before": 7, "frames_after": 2, "trimmed_start": 3,
        "trimmed_end": 2, "trimmed_interior": 0, "frames_removed": 5, "mode": "trim_edges"}
    assert episode.transform_results[rule.name] == result.to_dict()
    assert rule.apply(episode).changed is False


def test_gripper_records_actual_changes_even_when_skip_disabled():
    episode = work([[0, .2], [0, .8]])
    rule = GripperBinarizeRule(GripperBinarizeConfig(targets=["state.gripper"], skip_if_already_binary=False), dataset())
    assert rule.run(episode).metrics["changed_values"] == 2
    again = rule.run(episode)
    assert not again.changed and again.metrics["changed_values"] == 0


@pytest.mark.parametrize("kind", ["repair", "clip"])
def test_numeric_metrics_are_per_call_not_cumulative(kind):
    cfg = NumericSanityConfig(on_nan="clip", outlier_mode="clip_quantile")
    rule = NumericRepairRule(cfg, dataset()) if kind == "repair" else PercentileClipRule(cfg, dataset(), BOUNDS)
    first = rule.run(work([np.nan, 0] if kind == "repair" else [5, 0]))
    second = rule.run(work([0, 0]))
    assert first.changed and not second.changed
    assert second.metrics == {"frames_before": 2, "frames_after": 2}


def test_video_result_describes_deferred_instruction_changes():
    episode = work([0, 1])
    episode.ref.video_paths = {"cam": Path("unused.mp4")}
    rule = VideoRoiCropRule(VideoRoiCropConfig(rois={"cam": {
        "x_ratio": 0, "y_ratio": 0, "w_ratio": .5, "h_ratio": 1}}), dataset())
    assert rule.run(episode).changed
    result = rule.run(episode)
    assert not result.changed
    assert result.metrics["deferred_video_encoding"] is True


def test_replacement_episode_is_adopted_and_history_preserved():
    class Replace(TransformRule):
        name = "replace"
        def transform(self, episode, context=None):
            return TransformResult(work([9]), True, {"frames_after": 1})
    class Observe(TransformRule):
        name = "observe"
        def transform(self, episode, context=None):
            assert len(episode.df) == 1
            assert np.stack(episode.df["observation.state"])[0, 0] == 9
            return TransformResult(episode, False, {})
    episode = work([0, 1])
    episode.check_results["test"] = CheckResult(True, "test", "info")
    run_episode_stages(episode, [], [Replace(None, dataset()), Observe(None, dataset())])
    assert list(episode.transform_results) == ["replace", "observe"]
    assert "test" in episode.check_results
    json.dumps(episode.transform_results, allow_nan=False)
    assert "episode" not in episode.transform_results["replace"]


def test_dry_run_exports_transform_metrics_to_both_json_reports():
    episode = work([0, 0, 1, 2, 2])
    ref = SimpleNamespace(episode_index=18, tasks=[], video_paths={}, load_parquet=lambda: episode.df)
    rule = StaticFrameTrimRule(StaticFrameTrimConfig(pos_threshold=.01, rot_threshold_deg=None), dataset())
    report = CleaningReport(None, SimpleNamespace(to_dict=lambda: {}), [])
    report.write = Mock()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.source = SimpleNamespace(read_episode=lambda index: episode)
    pipeline.output = Path("unused-transform-report")
    pipeline._dry_run([ref], [], [rule], report)
    record = report.episode_quality[18]["transforms"][rule.name]
    assert record["changed"] and record["metrics"]["frames_after"] == 2
    opened = mock_open()
    with patch("builtins.open", opened):
        report._write_json(Path("report.json"), True)
    payload = json.loads("".join(c.args[0] for c in opened().write.call_args_list))
    assert payload["episode_modifications"][0]["transforms"][rule.name] == record


def test_none_return_is_rejected():
    class Bad(TransformRule):
        def transform(self, episode, context=None):
            return None
    with pytest.raises(TypeError, match="must return TransformResult"):
        Bad(None, dataset()).run(work([0, 1]))
