"""Legacy R6 finite/limit repairs and frame filtering, after all checks pass."""
import numpy as np

from lerobot_cleaner.v21.config import OnBad, OutlierMode
from lerobot_cleaner.v21.rules.base import TransformRule
from lerobot_cleaner.v21.rules.checks.trajectory._common import (
    COLUMNS,
    outlier_mask,
    stack,
    target_mask,
)
from lerobot_cleaner.v21.types import TransformResult


class NumericRepairRule(TransformRule):
    name = "numeric_repair"

    def __init__(self, config, dataset, outlier_bounds=None):
        super().__init__(config, dataset)
        self.outlier_bounds = outlier_bounds or {}

    def transform(self, work, context=None) -> TransformResult:
        frames_before = len(work.df)
        stats_before = self.stats.copy()
        for col, modality in COLUMNS:
            for label, policy, detector in (
                ("nan", self.config.on_nan, np.isnan),
                ("inf", self.config.on_inf, np.isinf),
            ):
                arr = stack(work, col)
                if arr is None:
                    break
                bad = detector(arr).any(axis=1)
                n = int(bad.sum())
                if not n:
                    continue
                if policy == OnBad.drop_frame:
                    work.restrict_to(~bad)
                    self.stats[f"{label}_frames_dropped"] += n
                elif policy == OnBad.clip:
                    arr[detector(arr)] = 0.0
                    work.df[col] = list(arr)
                    self.stats[f"{label}_frames_clipped"] += n
                elif policy == OnBad.interpolate:
                    # Component-wise: preserve every healthy value, even in a bad row.
                    component_mask = detector(arr)
                    x = np.arange(len(arr))
                    for j in range(arr.shape[1]):
                        missing = component_mask[:, j]
                        if not missing.any():
                            continue
                        good = np.isfinite(arr[:, j])
                        arr[missing, j] = np.interp(x[missing], x[good], arr[good, j]) if good.any() else 0.0
                    self.stats[f"{label}_components_interpolated"] += int(component_mask.sum())
                    work.df[col] = list(arr)
                    self.stats[f"{label}_frames_interpolated"] += n
            for dotted, (low, high) in self.config.joint_limits.items():
                arr = stack(work, col)
                if arr is None or dotted.split(".", 1)[0] != modality:
                    continue
                sl = self.resolver.resolve(dotted)
                sub = arr[:, sl.start:sl.end]
                bad = (np.isfinite(sub) & ((sub < low) | (sub > high))).any(axis=1)
                n = int(bad.sum())
                if not n:
                    continue
                if self.config.on_limit_violation == OnBad.drop_frame:
                    work.restrict_to(~bad)
                    self.stats["limit_frames_dropped"] += n
                elif self.config.on_limit_violation == OnBad.clip:
                    arr[:, sl.start:sl.end] = np.where(np.isfinite(sub), np.clip(sub, low, high), sub)
                    work.df[col] = list(arr)
                    self.stats["limit_frames_clipped"] += n
            arr = stack(work, col)
            if arr is not None and self.config.outlier_mode == OutlierMode.drop_frame and modality in self.outlier_bounds:
                low, high = self.outlier_bounds[modality]
                targets = target_mask(arr, modality, self.config.outlier_targets, self.resolver)
                bad = outlier_mask(arr, low, high, targets).any(axis=1)
                if bad.any():
                    work.restrict_to(~bad)
                    self.stats[f"outlier_frames_dropped_{modality}"] += int(bad.sum())
        if not len(work.df):
            work.drop("numeric repair removed all frames")
            self.stats["episodes_dropped_empty"] += 1

        changes = dict(self.stats - stats_before)
        metrics = {"frames_before": frames_before, "frames_after": len(work.df), **changes}
        changed = any(count for key, count in changes.items() if not key.startswith("episodes_"))
        return TransformResult(work, changed, metrics)
