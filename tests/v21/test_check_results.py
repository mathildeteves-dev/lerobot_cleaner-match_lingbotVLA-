"""Quality payloads survive checks, workers, dry-run, and strict JSONL export."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
from test_numeric_stages import dataset, work

from lerobot_cleaner.v21.config import DerivativeCheckConfig, EpisodeLengthFilterConfig
from lerobot_cleaner.v21.parallel import process_episode
from lerobot_cleaner.v21.pipeline import Pipeline
from lerobot_cleaner.v21.report import CleaningReport
from lerobot_cleaner.v21.rules.base import CheckResult, CheckRule
from lerobot_cleaner.v21.rules.checks.integrity.episode_length import EpisodeLengthFilterRule
from lerobot_cleaner.v21.rules.checks.trajectory.jerk import JerkRule


@pytest.mark.parametrize("policy,severity,dropped", [("warn_only", "warning", False), ("strict_drop", "error", True)])
def test_jerk_result_keeps_measurement_separate_from_acceptance(policy, severity, dropped):
    t = np.arange(6) * .2
    episode = work(t**3, t)
    rule = JerkRule(DerivativeCheckConfig(enabled=True, limits={"state.arm": 5}, on_violation=policy), dataset())
    result = rule.run(episode)
    assert isinstance(result, CheckResult)
    assert result.passed is False and result.severity == severity
    assert episode.dropped is dropped
    assert result.metrics["max_jerk"] == pytest.approx(6)
    assert result.metrics["threshold"] == 5
    assert episode.check_results["jerk"] is result
    assert pickle.loads(pickle.dumps(result)) == result


def test_insufficient_data_is_not_reported_as_pass():
    result = JerkRule(DerivativeCheckConfig(enabled=True, limits={"state.arm": 5}), dataset()).run(work([0, 1]))
    assert not result.passed
    assert result.metrics["evaluated"] is False
    assert result.severity == "warning"


def test_successful_check_returns_measured_quality():
    result = EpisodeLengthFilterRule(EpisodeLengthFilterConfig(min_frames=2), dataset()).check(work([0, 1]))
    assert result.passed and result.severity == "info"
    assert result.metrics["frames"] == 2


def test_worker_and_dry_run_propagate_rejected_check_results():
    episode = work([0, 1])
    ref = SimpleNamespace(episode_index=18, tasks=[], video_paths={}, load_parquet=lambda: episode.df)
    check = EpisodeLengthFilterRule(EpisodeLengthFilterConfig(min_frames=3), dataset())
    result = process_episode(ref, [check], [], Path("unused-quality-staging"), "libx264", 10, [])
    assert result.dropped and not result.check_results[check.name].passed
    report = CleaningReport(None, None, [check.name])
    report.write = Mock()
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.source = SimpleNamespace(read_episode=lambda index: episode)
    pipeline.output = Path("unused-quality-dry-run")
    pipeline._dry_run([ref], [check], [], report)
    assert report.episode_quality[18]["checks"][check.name]["passed"] is False
    report.write.assert_called_once_with(pipeline.output / "cleaning_report", dry_run=True)


def test_quality_jsonl_is_sorted_strict_json_and_includes_rejections():
    report = CleaningReport(None, None, [])
    for index, dropped in [(18, True), (2, False)]:
        report.record_episode(SimpleNamespace(src_index=index, dropped=dropped,
            drop_reason="bad" if dropped else None, length=2, notes={}, transform_results={},
            check_results={"jerk": CheckResult(not dropped, "jerk", "warning" if dropped else "info",
                {"max_jerk": np.float64(31.2), "unavailable": float("nan"), "values": np.array([1, 2])})}))
    mocked = mock_open()
    with patch.object(Path, "open", mocked):
        report._write_episode_quality(Path("episode_quality.jsonl"))
    lines = "".join(call.args[0] for call in mocked().write.call_args_list).splitlines()
    records = [json.loads(line) for line in lines]
    assert [record["episode_index"] for record in records] == [2, 18]
    assert records[1]["dropped"] is True
    assert records[1]["checks"]["jerk"]["metrics"]["unavailable"] is None
    assert records[1]["checks"]["jerk"]["metrics"]["max_jerk"] == 31.2


def test_boolean_check_result_is_rejected():
    class BadCheck(CheckRule):
        def check(self, episode, context=None):
            return False
    with pytest.raises(TypeError, match="must return CheckResult"):
        BadCheck(None, dataset()).run(work([0, 1]))
