"""R3: gripper binarization.

Thresholds the configured gripper dims to a binary value. Idempotent: if the
target already looks binary (only two distinct values matching the target set),
the rule skips it so re-running the pipeline is safe.
"""

from __future__ import annotations

import numpy as np

from lerobot_cleaner.v21.config import GripperMode
from lerobot_cleaner.v21.reader import ACTION_COL, STATE_COL
from lerobot_cleaner.v21.rules.base import TransformRule
from lerobot_cleaner.v21.types import EpisodeWork, TransformResult


class GripperBinarizeRule(TransformRule):
    name = "gripper_binarize"

    def _values(self, mode: GripperMode) -> tuple[float, float]:
        if mode == GripperMode.binary_01:
            return 0.0, 1.0
        return -1.0, 1.0

    def transform(self, work: EpisodeWork, context=None) -> TransformResult:
        metrics = {"changed_values": 0, "targets": {}}
        if not len(work.df):
            return TransformResult(work, False, metrics)
        low_val, high_val = self._values(self.config.mode)
        target_set = {low_val, high_val}

        # Cache stacked arrays per column to avoid repeated stack/unstack.
        cache: dict[str, np.ndarray] = {}
        for dotted in self.config.targets:
            sl = self.resolver.resolve(dotted)
            col = STATE_COL if sl.modality == "state" else ACTION_COL
            if col not in work.df.columns:
                continue
            if col not in cache:
                cache[col] = np.stack(work.df[col].to_numpy()).astype(np.float64)
            arr = cache[col]
            sub = arr[:, sl.start : sl.end]

            if self.config.skip_if_already_binary:
                uniq = set(np.unique(sub).tolist())
                if uniq.issubset(target_set):
                    self.stats[f"skipped_already_binary:{dotted}"] += 1
                    metrics["targets"][dotted] = {"changed_values": 0, "skipped_already_binary": True}
                    continue

            thresh = self.config.per_target_threshold.get(dotted, self.config.threshold)
            values = np.where(sub > thresh, high_val, low_val)
            count = int(np.count_nonzero(sub != values))
            metrics["targets"][dotted] = {"changed_values": count, "threshold": thresh}
            metrics["changed_values"] += count
            arr[:, sl.start : sl.end] = values
            self.stats[f"binarized:{dotted}"] += 1

        for col, arr in cache.items():
            work.df[col] = list(arr)

        return TransformResult(work, metrics["changed_values"] > 0, metrics)
