"""Typed contracts for episode checks and transformations.

Checks may annotate/reject an episode but must not edit data. Transforms may
change data while preserving frame alignment. Context is optional per-run data;
configuration and dataset metadata remain on each rule instance. Rule instances
must remain picklable for worker processes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any

from lerobot_cleaner.v21.types import CheckResult, EpisodeWork, TransformResult

if TYPE_CHECKING:
    from lerobot_cleaner.v21.reader import LeRobotDataset


class BaseRule(ABC):
    name: str = "base"

    def __init__(self, config, dataset: LeRobotDataset):
        self.config = config
        self.dataset = dataset
        self.resolver = dataset.resolver
        self.fps = dataset.fps
        self.stats: Counter = Counter()

    @abstractmethod
    def run(self, episode: EpisodeWork, context: Any = None) -> CheckResult | TransformResult | None:
        """Execute this rule using optional per-run context."""

    def apply(self, work: EpisodeWork) -> CheckResult | TransformResult | None:
        """Compatibility entry point for callers using apply(work)."""
        return self.run(work)

    def merge_stats(self, other: Counter) -> None:
        self.stats.update(other)

    def report_summary(self) -> dict:
        return dict(self.stats)


class CheckRule(BaseRule):
    def run(self, episode: EpisodeWork, context: Any = None) -> CheckResult:
        result = self.check(episode, context)
        if not isinstance(result, CheckResult):
            raise TypeError(f"{self.name}.check() must return CheckResult")
        if result.rule != self.name:
            raise ValueError(f"CheckResult.rule must match {self.name}")
        episode.check_results[self.name] = result
        return result

    def result(self, work, passed: bool, metrics=None, message=None) -> CheckResult:
        return CheckResult(
            passed=passed, rule=self.name,
            severity="info" if passed else ("error" if work.dropped else "warning"),
            metrics={} if metrics is None else metrics,
            message=message or (work.drop_reason if work.dropped else None),
        )

    @abstractmethod
    def check(self, episode: EpisodeWork, context: Any = None) -> CheckResult:
        """Return measured quality; only notes, rejection state, and counters may change."""


class TransformRule(BaseRule):
    def run(self, episode: EpisodeWork, context: Any = None) -> TransformResult:
        result = self.transform(episode, context)
        if not isinstance(result, TransformResult):
            raise TypeError(f"{self.name}.transform() must return TransformResult")
        if not isinstance(result.episode, EpisodeWork):
            raise TypeError("TransformResult.episode must be EpisodeWork")
        if result.episode is not episode:
            # Preserve the existing mutable-work API while accepting replacement episodes.
            previous_checks = dict(episode.check_results)
            previous_transforms = dict(episode.transform_results)
            previous_notes = dict(episode.notes)
            episode.__dict__.update(result.episode.__dict__)
            episode.check_results = {**previous_checks, **episode.check_results}
            episode.transform_results = {**previous_transforms, **episode.transform_results}
            episode.notes = {**previous_notes, **episode.notes}
            result.episode = episode
        episode.transform_results[self.name] = result.to_dict()
        return result

    @abstractmethod
    def transform(self, episode: EpisodeWork, context: Any = None) -> TransformResult:
        """Return the resulting episode and explicit, per-call modification metrics."""


class Rule(BaseRule):
    """Legacy adapter for existing subclasses that implement only apply(work).

    New rules should inherit CheckRule or TransformRule. This adapter allows
    external apply-only subclasses to participate in run-based dispatch.
    """

    def run(self, episode: EpisodeWork, context: Any = None) -> None:
        return self.apply(episode)

    @abstractmethod
    def apply(self, work: EpisodeWork) -> None:
        """Legacy execution method."""
