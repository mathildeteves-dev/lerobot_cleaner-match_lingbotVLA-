"""Rule interfaces, compatibility, and context propagation."""
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from lerobot_cleaner.v21.config import CleaningConfig
from lerobot_cleaner.v21.rules import build_checks, build_transforms, run_episode_stages
from lerobot_cleaner.v21.rules.base import BaseRule, CheckResult, CheckRule, Rule, TransformRule
from lerobot_cleaner.v21.types import EpisodeWork, TransformResult


class RecordingCheck(CheckRule):
    name = "recording_check"

    def check(self, episode, context=None):
        episode.calls.append((self.name, context))
        self.stats["called"] += 1
        return CheckResult(True, self.name, "info", {})


class RecordingTransform(TransformRule):
    name = "recording_transform"

    def transform(self, episode, context=None):
        episode.calls.append((self.name, context))
        self.stats["called"] += 1
        return TransformResult(episode, False, {})


def make_episode():
    episode = EpisodeWork(ref=None, df=pd.DataFrame(), keep_indices=[])
    episode.calls = []
    return episode


class LegacyRule(Rule):
    def apply(self, work):
        work.calls.append("legacy")


def dataset():
    return SimpleNamespace(resolver=None, fps=10)


@pytest.mark.parametrize("cls", [BaseRule, CheckRule, TransformRule, Rule])
def test_abstract_interfaces_require_an_implementation(cls):
    with pytest.raises(TypeError):
        cls(None, dataset())


@pytest.mark.parametrize("cls", [RecordingCheck, RecordingTransform])
def test_run_dispatches_context_and_apply_remains_compatible(cls):
    rule = cls(None, dataset())
    episode = make_episode()
    context = object()
    rule.run(episode, context)
    rule.apply(episode)
    assert episode.calls == [(rule.name, context), (rule.name, None)]
    assert rule.report_summary() == {"called": 2}


def test_stage_runner_propagates_context_to_both_interfaces():
    episode = make_episode()
    context = object()
    checks = [RecordingCheck(None, dataset())]
    transforms = [RecordingTransform(None, dataset())]
    stats = run_episode_stages(episode, checks, transforms, context)
    assert episode.calls == [("recording_check", context), ("recording_transform", context)]
    assert stats["recording_check"]["called"] == 1
    assert stats["recording_transform"]["called"] == 1


def test_apply_only_external_rule_supports_run():
    episode = make_episode()
    LegacyRule(None, dataset()).run(episode, {})
    assert episode.calls == ["legacy"]


def test_builders_return_typed_picklable_rules():
    cfg = CleaningConfig(rules={
        "timestamp_alignment": {"enabled": True},
        "episode_length_filter": {"enabled": True},
        "video_integrity": {"enabled": True},
        "static_frame_trim": {"enabled": True},
        "gripper_binarize": {"enabled": True},
        "video_roi_crop": {"enabled": True},
        "numeric_sanity": {"enabled": True, "on_nan": "interpolate",
            "joint_limits": {"state.arm": [-1, 1]}, "outlier_mode": "clip_quantile",
            "velocity": {"enabled": True, "limits": {"state.arm": 1}},
            "acceleration": {"enabled": True, "limits": {"state.arm": 1}},
            "jerk": {"enabled": True, "limits": {"state.arm": 1}}},
    })
    checks = build_checks(cfg, dataset())
    transforms = build_transforms(cfg, dataset())
    assert len(checks) == 9 and len(transforms) == 5
    assert all(isinstance(rule, CheckRule) for rule in checks)
    assert all(isinstance(rule, TransformRule) for rule in transforms)
    restored = pickle.loads(pickle.dumps(checks + transforms))
    assert [type(rule) for rule in restored] == [type(rule) for rule in checks + transforms]
