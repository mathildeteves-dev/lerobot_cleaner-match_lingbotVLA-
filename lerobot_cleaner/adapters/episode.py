"""Shared data structures passed between rules, pipeline, and writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lerobot_cleaner.core.trajectory import CheckResult, TrajectoryView, _json_safe


@dataclass
class EpisodeRef:
    episode_index: int
    length: int
    tasks: list[str]
    parquet_path: Path | None
    video_paths: dict[str, Path] = field(default_factory=dict)

    def load_parquet(self):
        if self.parquet_path is None:
            raise ValueError("Sharded episodes must be read through their DatasetAdapter")
        from lerobot_cleaner.v21.legacy_reader import load_legacy_parquet
        return load_legacy_parquet(self.parquet_path)



@dataclass
class VideoTransform:
    """How to materialize one video key for the cleaned episode."""

    # 0-based source frame indices to keep, in order. None => keep all rows.
    keep_indices: Optional[list[int]] = None
    # (x_ratio, y_ratio, w_ratio, h_ratio) relative crop, or None.
    crop_ratio: Optional[tuple[float, float, float, float]] = None
    # (width, height) to resize to after crop, or None.
    resize_wh: Optional[tuple[int, int]] = None


@dataclass
class UnifiedEpisode:
    """Mutable working state for one episode as it flows through the rules.

    Rules read/modify ``df`` (the parquet rows) and may set ``keep_indices``
    (which original frame rows survive) and ``video_transforms`` (per-key crop/
    resize). ``dropped`` short-circuits the rest of the pipeline.
    """

    ref: EpisodeRef
    df: pd.DataFrame
    # Original row indices (0-based into the source parquet/video) that survive.
    # Always kept in sync with df: len(keep_indices) == len(df).
    keep_indices: list[int]
    video_transforms: dict[str, VideoTransform] = field(default_factory=dict)
    dropped: bool = False
    drop_reason: Optional[str] = None
    # Per-rule notes for the report: rule_name -> message.
    notes: dict[str, str] = field(default_factory=dict)

    check_results: dict[str, CheckResult] = field(default_factory=dict)
    transform_results: dict[str, dict] = field(default_factory=dict)

    fps: float | None = None
    state_features: tuple | None = None
    action_features: tuple | None = None
    camera_features: tuple = ()
    metadata: dict = field(default_factory=dict)
    feature_schema: object | None = None
    assembly_policy: object | None = None

    @property
    def episode_index(self):
        return self.ref.episode_index

    def to_trajectory(self, fps=None, state_column="observation.state", action_column="action", *, include_padding=False):
        fps = self.fps if fps is None else fps
        if fps is None:
            raise ValueError("Episode fps is required")
        from .builder import EpisodeBuilder
        from .schema import CanonicalFeatureSchema
        schema = self.feature_schema or CanonicalFeatureSchema(
            states=self.state_features or (), actions=self.action_features or (),
            raw_vectors=self.state_features is None and self.action_features is None,
            state_column=state_column, action_column=action_column)
        frame = self.df
        indices = np.asarray(self.keep_indices)
        valid = self.metadata.get("assembly", {}).get("valid_mask")
        if valid is not None and not include_padding:
            selected = np.asarray(valid, dtype=bool)[indices]
            frame = frame.iloc[np.flatnonzero(selected)]
            indices = indices[selected]
        state, action = EpisodeBuilder(schema, fps, self.assembly_policy).arrays(frame)
        timestamps = (frame["timestamp"].to_numpy(dtype=float) if "timestamp" in frame
                      else indices.astype(float) / fps)
        return TrajectoryView(state, action, timestamps, fps)

    def drop(self, reason: str) -> None:
        self.dropped = True
        self.drop_reason = reason

    def note(self, rule: str, msg: str) -> None:
        self.notes[rule] = msg

    def restrict_to(self, mask_or_indices) -> None:
        """Keep only the given rows (boolean mask or positional index list)
        and update keep_indices + df consistently."""
        import numpy as np

        arr = np.asarray(mask_or_indices)
        if arr.dtype == bool:
            positions = np.flatnonzero(arr)
        else:
            positions = arr.astype(int)
        self.df = self.df.iloc[positions].reset_index(drop=True)
        self.keep_indices = [self.keep_indices[p] for p in positions]


@dataclass
class TransformResult:
    """Modification result; only the lightweight summary is stored in reports."""

    episode: UnifiedEpisode
    changed: bool
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Never serialize the episode/DataFrame, or create a circular result history.
        return _json_safe({"changed": self.changed, "metrics": self.metrics})
