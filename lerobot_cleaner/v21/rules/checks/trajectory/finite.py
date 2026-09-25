"""Detect NaN/Inf without modifying rows or values."""
from lerobot_cleaner.core.quality import check_finite
from lerobot_cleaner.v21.adapter import V21Adapter
from lerobot_cleaner.v21.config import OnBad
from lerobot_cleaner.v21.rules.base import CheckRule

from ._common import COLUMNS


class FiniteRule(CheckRule):
    name = "finite"

    def check(self, work, context=None):
        metrics = {}
        problems = []
        measured = check_finite(V21Adapter.to_trajectory(work, self.fps))
        for col, modality in COLUMNS:
            for label, action in (("nan", self.config.on_nan), ("inf", self.config.on_inf)):
                n = measured.metrics[modality][f"{label}_frames"]
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
