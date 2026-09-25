"""Clip selected finite values to dataset percentile bounds."""
import numpy as np

from lerobot_cleaner.v21.rules.base import TransformRule
from lerobot_cleaner.v21.rules.checks.trajectory._common import (
    COLUMNS,
    outlier_mask,
    stack,
    target_mask,
)
from lerobot_cleaner.v21.types import TransformResult


class PercentileClipRule(TransformRule):
    name = "percentile_clip"

    def __init__(self, config, dataset, outlier_bounds=None):
        super().__init__(config, dataset)
        self.outlier_bounds = outlier_bounds or {}

    def transform(self, work, context=None) -> TransformResult:
        frames_before = len(work.df)
        stats_before = self.stats.copy()
        for col, modality in COLUMNS:
            arr = stack(work, col)
            if arr is None or modality not in self.outlier_bounds:
                continue
            low, high = self.outlier_bounds[modality]
            targets = target_mask(arr, modality, self.config.outlier_targets, self.resolver)
            mask = outlier_mask(arr, low, high, targets)
            n = int(mask.any(axis=1).sum())
            if n:
                work.df[col] = list(np.where(mask, np.clip(arr, low, high), arr))
                self.stats[f"outlier_frames_clipped_{modality}"] += n

        changes = dict(self.stats - stats_before)
        metrics = {"frames_before": frames_before, "frames_after": len(work.df), **changes}
        changed = any(count for key, count in changes.items() if not key.startswith("episodes_"))
        return TransformResult(work, changed, metrics)
