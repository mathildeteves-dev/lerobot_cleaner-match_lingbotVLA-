"""Shared data structures passed between rules, pipeline, and writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

from lerobot_cleaner.v21.reader import EpisodeRef


def _json_safe(value):
    import math
    from numbers import Integral, Real

    def convert(value):
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            return float(value) if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if hasattr(value, "tolist"):
            return convert(value.tolist())
        raise TypeError(f"Unsupported check metric type: {type(value).__name__}")

    return convert(value)


@dataclass
class CheckResult:
    """An observed quality result, independent of the episode acceptance policy."""

    passed: bool
    rule: str
    severity: str
    metrics: dict = field(default_factory=dict)
    message: str | None = None

    def to_dict(self) -> dict:
        """Return strict JSON-compatible data; unavailable/non-finite numbers are null."""
        return _json_safe(asdict(self))


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
class EpisodeWork:
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

    episode: EpisodeWork
    changed: bool
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Never serialize the episode/DataFrame, or create a circular result history.
        return _json_safe({"changed": self.changed, "metrics": self.metrics})
