"""R1: timestamp & frame alignment checks.

Verifies the source episode is internally consistent before other rules run:
  - timestamp monotonic increasing
  - mean dt within tolerance of 1/fps
  - inter-frame dt uniform (GR00T assumes dt == 1/fps for action chunking)
  - video frame count within tolerance of parquet row count
"""

from __future__ import annotations

import numpy as np

from lerobot_cleaner.v21.config import OnMismatch
from lerobot_cleaner.v21.rules.base import CheckRule
from lerobot_cleaner.v21.types import CheckResult, EpisodeWork
from lerobot_cleaner.v21.video_utils import VideoError, count_frames


class TimestampAlignmentRule(CheckRule):
    name = "timestamp_alignment"

    def check(self, work: EpisodeWork, context=None) -> CheckResult:
        df = work.df
        problems: list[str] = []
        metrics = {"row_count": len(df), "video_frame_counts": {}}

        if "timestamp" in df.columns and len(df) > 1:
            ts = df["timestamp"].to_numpy(dtype=np.float64)
            dt = np.diff(ts)
            metrics.update(mean_dt=float(np.mean(dt)), min_dt=float(np.min(dt)), max_dt=float(np.max(dt)))
            if not np.isfinite(ts).all():
                problems.append("non-finite timestamp")
            if np.any(dt <= 0):
                problems.append("non-monotonic timestamp")
            expected_dt = 1.0 / self.fps if self.fps else None
            if expected_dt:
                mean_dt = float(np.mean(dt))
                ratio = abs(mean_dt - expected_dt) / expected_dt
                if ratio > self.config.expected_fps_tolerance_ratio:
                    problems.append(
                        f"mean dt {mean_dt:.4f}s deviates {ratio:.0%} from 1/fps {expected_dt:.4f}s"
                    )
                if self.config.require_uniform_dt:
                    # Coefficient of variation of dt as a uniformity proxy.
                    if mean_dt > 0 and float(np.std(dt)) / mean_dt > 0.25:
                        problems.append("non-uniform inter-frame dt (action chunking assumes uniform)")

        # Video frame count vs rows.
        n_rows = len(df)
        for vkey, vpath in work.ref.video_paths.items():
            try:
                frames = count_frames(vpath)
            except VideoError:
                continue  # R7 handles decodability
            metrics["video_frame_counts"][vkey] = frames
            if abs(frames - n_rows) > self.config.video_frame_tolerance:
                problems.append(f"video {vkey}: {frames} frames vs {n_rows} rows")

        if problems:
            self.stats["episodes_with_mismatch"] += 1
            msg = "; ".join(problems)
            if self.config.on_mismatch == OnMismatch.strict_drop:
                work.drop(f"timestamp/alignment: {msg}")
                self.stats["episodes_dropped"] += 1
            else:
                work.note(self.name, msg)

        return self.result(work, not problems, metrics, "; ".join(problems) or None)
