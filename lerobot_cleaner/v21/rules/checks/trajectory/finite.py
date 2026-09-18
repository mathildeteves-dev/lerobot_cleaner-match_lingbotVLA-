"""Detect NaN/Inf without modifying rows or values."""
import numpy as np

from lerobot_cleaner.v21.config import OnBad
from lerobot_cleaner.v21.rules.base import CheckRule

from ._common import COLUMNS, stack


class FiniteRule(CheckRule):
    name = "finite"

    def check(self, work, context=None):
        metrics = {}
        problems = []
        for col, _ in COLUMNS:
            arr = stack(work, col)
            if arr is None:
                continue
            for label, mask, action in (
                ("nan", np.isnan(arr).any(axis=1), self.config.on_nan),
                ("inf", np.isinf(arr).any(axis=1), self.config.on_inf),
            ):
                n = int(mask.sum())
                metrics.setdefault(col, {})[f"{label}_frames"] = n
                if not n:
                    continue
                problems.append(f"{n} {label} frames in {col}")
                self.stats[f"{label}_frames_detected"] += n
                work.note(f"{self.name}:{col}:{label}", f"{n} {label} frames; policy={action.value}")
                if action == OnBad.drop_episode:
                    work.drop(f"{label} in {col}")
                    self.stats[f"episodes_dropped_{label}"] += 1
                    return self.result(work, False, metrics, "; ".join(problems))

        return self.result(work, not problems, metrics, "; ".join(problems) or None)
