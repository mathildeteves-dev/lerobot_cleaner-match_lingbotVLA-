"""Rule base class and registry.

A Rule is a pluggable, stateless-per-episode transform. Each rule:
  - declares its config dataclass (a RuleConfig subclass)
  - implements ``apply(work)`` mutating the EpisodeWork in place
  - accumulates counters into ``self.stats`` for the report

Rules run inside worker processes, so they must not hold un-picklable state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

from lerobot_cleaner.types import EpisodeWork

if TYPE_CHECKING:
    from lerobot_cleaner.dataset.reader import LeRobotDataset


class Rule(ABC):
    #: stable rule name matching the config key
    name: str = "base"

    def __init__(self, config, dataset: "LeRobotDataset"):
        self.config = config
        self.dataset = dataset
        self.resolver = dataset.resolver
        self.fps = dataset.fps
        self.stats: Counter = Counter()

    @abstractmethod
    def apply(self, work: EpisodeWork) -> None:
        """Mutate ``work`` in place. Set ``work.drop(...)`` to drop an episode."""

    def merge_stats(self, other: Counter) -> None:
        self.stats.update(other)

    def report_summary(self) -> dict:
        return dict(self.stats)
