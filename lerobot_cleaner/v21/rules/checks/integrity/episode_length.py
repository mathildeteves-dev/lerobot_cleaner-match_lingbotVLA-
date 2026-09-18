"""R5: episode length filter — drop episodes that are too short/long."""

from __future__ import annotations

from lerobot_cleaner.v21.rules.base import CheckRule
from lerobot_cleaner.v21.types import CheckResult, EpisodeWork


class EpisodeLengthFilterRule(CheckRule):
    name = "episode_length_filter"

    def check(self, work: EpisodeWork, context=None) -> CheckResult:
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

        return self.result(work, reason is None, {
            "frames": n, "duration_sec": duration, "min_frames": cfg.min_frames,
            "max_frames": cfg.max_frames, "min_duration_sec": cfg.min_duration_sec,
            "max_duration_sec": cfg.max_duration_sec,
        }, reason)
