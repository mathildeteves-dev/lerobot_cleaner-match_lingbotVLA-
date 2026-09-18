"""R2: static frame trimming.

Two modes:
  - trim_edges (default, safe): drop only the leading/trailing static run. Keeps
    interior sampling uniform, so action chunking / dt stays correct.
  - drop_static_frames (opt-in, less safe): drop every static frame. This makes
    inter-row dt non-uniform; GR00T builds action chunks from consecutive rows
    assuming dt == 1/fps, so the chunk's wall-clock span becomes variable. We
    therefore emit a loud warning and the writer rebuilds a uniform timestamp.
"""

from __future__ import annotations

import numpy as np

from lerobot_cleaner.v21.config import StaticSource, StaticTrimMode
from lerobot_cleaner.v21.reader import ACTION_COL, STATE_COL
from lerobot_cleaner.v21.rules.base import TransformRule
from lerobot_cleaner.v21.types import EpisodeWork, TransformResult


class StaticFrameTrimRule(TransformRule):
    name = "static_frame_trim"

    def _source_array(self, work: EpisodeWork) -> np.ndarray:
        col = STATE_COL if self.config.source == StaticSource.state else ACTION_COL
        return np.stack(work.df[col].to_numpy()).astype(np.float64)

    def _motion(self, arr: np.ndarray) -> np.ndarray:
        """Per-step motion magnitude (length n; first element is 0)."""
        if len(arr) <= 1:
            return np.zeros(len(arr))
        diff = np.abs(np.diff(arr, axis=0))

        rot_cols = self._rotation_columns(arr.shape[1])
        pos_cols = np.array([c for c in range(arr.shape[1]) if c not in rot_cols])

        per_step = np.zeros(len(arr) - 1)
        if pos_cols.size:
            per_step = np.maximum(per_step, diff[:, pos_cols].max(axis=1))
        # Rotation handled separately with its own threshold (see is_static).
        self._last_rot_motion = (
            diff[:, list(rot_cols)].max(axis=1) if rot_cols else np.zeros(len(arr) - 1)
        )
        return np.concatenate([[0.0], per_step])

    def _rotation_columns(self, dim: int) -> set[int]:
        cols: set[int] = set()
        for dotted in self.config.rotation_keys:
            try:
                sl = self.resolver.resolve(dotted)
            except (KeyError, ValueError):
                continue
            cols.update(range(sl.start, sl.end))
        return {c for c in cols if c < dim}

    def _static_mask(self, arr: np.ndarray) -> np.ndarray:
        pos_motion = self._motion(arr)
        static = pos_motion < self.config.pos_threshold
        if self.config.rot_threshold_deg is not None and hasattr(self, "_last_rot_motion"):
            rot_thresh = np.deg2rad(self.config.rot_threshold_deg)
            rot_motion = np.concatenate([[0.0], self._last_rot_motion])
            static = static & (rot_motion < rot_thresh)
        return static

    def transform(self, work: EpisodeWork, context=None) -> TransformResult:
        before = list(work.keep_indices)
        if len(work.df) > self.config.min_kept_frames:
            arr = self._source_array(work)
            static = self._static_mask(arr)
            if self.config.mode == StaticTrimMode.trim_edges:
                self._trim_edges(work, static)
            else:
                self._drop_all_static(work, static)
        # Count positions in this transform's input, not gaps in source frame IDs.
        after = work.keep_indices
        removed = len(before) - len(after)
        start = before.index(after[0]) if after else len(before)
        end = len(before) - 1 - before.index(after[-1]) if after else 0
        return TransformResult(work, removed > 0, {
            "frames_before": len(before), "frames_after": len(after),
            "trimmed_start": start, "trimmed_end": end,
            "trimmed_interior": removed - start - end,
            "frames_removed": removed, "mode": self.config.mode.value,
        })

    def _trim_edges(self, work: EpisodeWork, static: np.ndarray) -> None:
        n = len(static)
        start = 0
        while start < n and static[start]:
            start += 1
        end = n
        while end > start and static[end - 1]:
            end -= 1
        # Guard minimum length.
        if end - start < self.config.min_kept_frames:
            work.note(self.name, "edge trim skipped (would go below min_kept_frames)")
            return
        trimmed = (start) + (n - end)
        if trimmed == 0:
            return
        work.restrict_to(np.arange(start, end))
        self.stats["edge_frames_trimmed"] += trimmed
        self.stats["episodes_trimmed"] += 1

    def _drop_all_static(self, work: EpisodeWork, static: np.ndarray) -> None:
        keep = ~static
        # Always keep the first frame as an anchor.
        keep[0] = True
        n_drop = int((~keep).sum())
        if n_drop == 0:
            return
        if int(keep.sum()) < self.config.min_kept_frames:
            work.note(self.name, "drop_static skipped (would go below min_kept_frames)")
            return
        work.restrict_to(keep)
        self.stats["static_frames_dropped"] += n_drop
        self.stats["episodes_modified"] += 1
        work.note(
            self.name,
            f"dropped {n_drop} interior static frames; inter-frame dt is now "
            f"non-uniform (timestamps rebuilt to uniform fps clock by writer).",
        )
