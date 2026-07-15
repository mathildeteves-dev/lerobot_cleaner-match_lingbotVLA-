"""R5: episode length filter — drop episodes that are too short/long."""

from __future__ import annotations

from lerobot_cleaner.rules.base import Rule
from lerobot_cleaner.types import EpisodeWork


class EpisodeLengthFilterRule(Rule):
    name = "episode_length_filter"

    def apply(self, work: EpisodeWork) -> None:
        n = len(work.df)
        cfg = self.config
        duration = n / self.fps if self.fps else 0.0

        reason = None
        if n < cfg.min_frames:
            reason = f"length {n} < min_frames {cfg.min_frames}"
        elif cfg.max_frames is not None and n > cfg.max_frames:
            reason = f"length {n} > max_frames {cfg.max_frames}"
        elif cfg.min_duration_sec is not None and duration < cfg.min_duration_sec:
            reason = f"duration {duration:.2f}s < min {cfg.min_duration_sec}s"
        elif cfg.max_duration_sec is not None and duration > cfg.max_duration_sec:
            reason = f"duration {duration:.2f}s > max {cfg.max_duration_sec}s"

        if reason:
            work.drop(reason)
            self.stats["episodes_dropped"] += 1
