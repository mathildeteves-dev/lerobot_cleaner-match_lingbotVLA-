"""Construct the three explicit cleaning stages (independent of R numbers)."""

from __future__ import annotations

from collections import Counter

from lerobot_cleaner.v21.rules.base import BaseRule, CheckRule, Rule, TransformRule
from lerobot_cleaner.v21.rules.checks.integrity.episode_length import EpisodeLengthFilterRule
from lerobot_cleaner.v21.rules.checks.integrity.timestamp import TimestampAlignmentRule
from lerobot_cleaner.v21.rules.checks.vision.video_integrity import VideoIntegrityRule
from lerobot_cleaner.v21.rules.finalizers import DatasetFinalizer, dataset_finalizers
from lerobot_cleaner.v21.rules.numeric import numeric_checks, numeric_transforms
from lerobot_cleaner.v21.rules.transforms.embodiment.gripper import GripperBinarizeRule
from lerobot_cleaner.v21.rules.transforms.motion.static_trim import StaticFrameTrimRule
from lerobot_cleaner.v21.rules.transforms.vision.roi_crop import VideoRoiCropRule
from lerobot_cleaner.v21.types import CheckResult, EpisodeWork, TransformResult

CHECKS = (
    TimestampAlignmentRule,
    EpisodeLengthFilterRule,
    VideoIntegrityRule,
)
TRANSFORMS = (StaticFrameTrimRule, GripperBinarizeRule, VideoRoiCropRule)

__all__ = ["TransformResult", "CheckResult", "BaseRule", "CheckRule", "TransformRule", "Rule", "build_checks", "build_transforms", "build_finalizers", "run_episode_stages"]


def build_checks(config, dataset, outlier_bounds=None) -> list[CheckRule]:
    """Build checks that may reject/annotate but never change episode data."""
    checks = []
    for cls in CHECKS:
        rule_cfg = getattr(config.rules, cls.name)
        if not rule_cfg.enabled:
            continue
        checks.append(cls(rule_cfg, dataset))
    if config.rules.numeric_sanity.enabled:
        checks.extend(numeric_checks(config.rules.numeric_sanity, dataset, outlier_bounds))
    return checks


def build_transforms(config, dataset, outlier_bounds=None) -> list[TransformRule]:
    """Build transformations for episodes accepted by every check."""
    repairs = numeric_transforms(config.rules.numeric_sanity, dataset, outlier_bounds) if config.rules.numeric_sanity.enabled else []
    return repairs + [
        cls(getattr(config.rules, cls.name), dataset)
        for cls in TRANSFORMS
        if getattr(config.rules, cls.name).enabled
    ]


def build_finalizers(config, lingbot_normalizer=None) -> list[DatasetFinalizer]:
    """Dataset finalization is mandatory, even when all episodes are rejected."""
    return dataset_finalizers(config.rules.reindex_and_restats, lingbot_normalizer)


def _run_stage(work: EpisodeWork, rules: list[BaseRule], stats: dict, context=None) -> None:
    for rule in rules:
        if work.dropped:
            return
        rule.stats = Counter()
        rule.run(work, context)
        stats[rule.name] = Counter(rule.stats)


def run_episode_stages(
    work: EpisodeWork, checks: list[CheckRule], transforms: list[TransformRule], context=None
) -> dict[str, Counter]:
    """Shared by workers and dry-run: reject early, then transform accepted data."""
    stats: dict[str, Counter] = {}
    _run_stage(work, checks, stats, context)
    if work.dropped:
        return stats
    _run_stage(work, transforms, stats, context)
    return stats
