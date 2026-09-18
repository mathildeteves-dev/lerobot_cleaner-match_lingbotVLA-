"""Detect percentile outliers; all repairs run in transforms."""
from lerobot_cleaner.v21.rules.base import CheckRule

from ._common import COLUMNS, outlier_mask, stack, target_mask


class OutlierRule(CheckRule):
    name = "outlier"

    def __init__(self, config, dataset, outlier_bounds=None):
        super().__init__(config, dataset)
        self.outlier_bounds = outlier_bounds or {}

    def check(self, work, context=None):
        metrics = {}
        problems = []
        for col, modality in COLUMNS:
            arr = stack(work, col)
            if arr is None:
                continue
            if modality not in self.outlier_bounds:
                problems.append(f"{col}: percentile bounds unavailable")
                metrics[modality] = {"evaluated": False}
                continue
            low, high = self.outlier_bounds[modality]
            targets = target_mask(arr, modality, self.config.outlier_targets, self.resolver)
            n = int(outlier_mask(arr, low, high, targets).any(axis=1).sum())
            metrics[modality] = {"evaluated": True, "outlier_frames": n, "low": low, "high": high}
            if n:
                problems.append(f"{col}: {n} percentile outlier frames")
                self.stats[f"outlier_frames_{modality}"] += n
                work.note(f"{self.name}:{modality}", f"{n} percentile outlier frames")

        if not metrics:
            problems.append("no numeric data available for outlier detection")
        return self.result(work, not problems, metrics, "; ".join(problems) or None)
