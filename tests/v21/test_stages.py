"""Stage boundaries, rejection, and mandatory dataset finalization."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from lerobot_cleaner.v21.config import CleaningConfig
from lerobot_cleaner.v21.parallel import process_episode
from lerobot_cleaner.v21.pipeline import Pipeline
from lerobot_cleaner.v21.rules import (
    build_checks,
    build_finalizers,
    build_transforms,
    run_episode_stages,
)
from lerobot_cleaner.v21.types import EpisodeWork


def make_ref():
    return SimpleNamespace(
        episode_index=0, tasks=[], video_paths={},
        load_parquet=lambda: pd.DataFrame({"timestamp": [0.0, 0.1]}),
    )


def make_rule(name, calls, reject=False):
    rule = SimpleNamespace(name=name, stats=Counter())

    def run(work, context=None):
        calls.append(name)
        rule.stats["visited"] += 1
        if reject:
            work.drop(name)

    rule.run = run
    return rule


def test_build_stage_membership_and_disabled_rules():
    cfg = CleaningConfig()
    ds = SimpleNamespace(resolver=None, fps=10)
    for name in type(cfg.rules).model_fields:
        getattr(cfg.rules, name).enabled = True
    assert [r.name for r in build_checks(cfg, ds)] == [
        "timestamp_alignment", "episode_length_filter", "video_integrity", "finite", "outlier",
    ]
    assert [r.name for r in build_transforms(cfg, ds)] == [
        "static_frame_trim", "gripper_binarize", "video_roi_crop",
    ]
    cfg.rules.numeric_sanity.enabled = False
    cfg.rules.static_frame_trim.enabled = False
    assert "finite" not in [r.name for r in build_checks(cfg, ds)]
    assert "static_frame_trim" not in [r.name for r in build_transforms(cfg, ds)]


@pytest.mark.parametrize("reject", [False, True])
def test_checks_gate_transforms_and_reset_counters(reject):
    calls = []
    checks = [make_rule("check", calls, reject)]
    transforms = [make_rule("transform", calls)]
    for _ in range(2):
        ref = make_ref()
        work = EpisodeWork(ref=ref, df=ref.load_parquet(), keep_indices=[0, 1])
        stats = run_episode_stages(work, checks, transforms)
        assert stats["check"]["visited"] == 1
        assert ("transform" in stats) is not reject
    assert calls == (["check"] if reject else ["check", "transform"]) * 2


def test_worker_rejection_never_stages_artifacts():
    calls = []
    result = process_episode(
        make_ref(), [make_rule("reject", calls, True), make_rule("later", calls)],
        [make_rule("transform", calls)], Path("unused-staging"), "libx264", 10, [],
    )
    assert calls == ["reject"]
    assert result.dropped
    assert result.staged_parquet is None
    assert result.staged_videos == {}


def test_dry_run_uses_same_acceptance_gate():
    calls = []
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.output = Path("unused-dry-run")
    ref = make_ref()
    pipeline.source = SimpleNamespace(read_episode=lambda index: EpisodeWork(
        ref=ref, df=ref.load_parquet(), keep_indices=[0, 1]))
    report = Mock()
    pipeline._dry_run(
        [make_ref()], [make_rule("reject", calls, True)],
        [make_rule("transform", calls)], report,
    )
    assert calls == ["reject"]
    assert report.record_episode.call_args.args[0].dropped
    report.write.assert_called_once_with(pipeline.output / "cleaning_report", dry_run=True)


def test_finalizer_always_writes_meta_even_without_accepted_episodes():
    cfg = CleaningConfig()
    cfg.rules.reindex_and_restats.enabled = False
    finalizers = build_finalizers(cfg)
    assert len(finalizers) == 3
    writer, report = Mock(), Mock()
    writer.episodes_out = []
    for finalizer in finalizers:
        finalizer.finalize([], writer, report)
    writer.finalize_metadata.assert_called_once_with()
    writer.rebuild_stats.assert_called_once_with()
    writer.finalize_stats.assert_called_once_with()
    writer.finalize_staged_episode.assert_not_called()


def test_finalizer_sorts_reindexes_and_skips_rejected_results():
    cfg = CleaningConfig()
    writer, report = Mock(), Mock()
    writer.verify_episode_alignment.return_value = None
    results = [
        SimpleNamespace(src_index=i, dropped=(i == 1), staged_parquet=str(i),
                        staged_videos={}, tasks=[], length=2)
        for i in [2, 1, 0]
    ]
    writer.episodes_out = [{"episode_index": 0, "length": 2}, {"episode_index": 1, "length": 2}]
    for finalizer in build_finalizers(cfg):
        finalizer.finalize(results, writer, report)
    assert [call.args[:2] for call in writer.finalize_staged_episode.call_args_list] == [
        (0, "0"), (1, "2"),
    ]
    assert report.record_episode.call_count == 3
    assert writer.verify_episode_alignment.call_count == 2
    writer.finalize_metadata.assert_called_once_with()


def test_length_check_applies_before_transform():
    cfg = CleaningConfig()
    for name in type(cfg.rules).model_fields:
        getattr(cfg.rules, name).enabled = False
    cfg.rules.episode_length_filter.enabled = True
    cfg.rules.episode_length_filter.min_frames = 2
    checks = build_checks(cfg, SimpleNamespace(resolver=None, fps=10))
    transform = SimpleNamespace(name="trim", stats=Counter())
    transform.run = lambda work, context=None: work.restrict_to([True, False])
    ref = make_ref()
    work = EpisodeWork(ref=ref, df=ref.load_parquet(), keep_indices=[0, 1])
    run_episode_stages(work, checks, [transform])
    assert not work.dropped
    assert len(work.df) == 1
