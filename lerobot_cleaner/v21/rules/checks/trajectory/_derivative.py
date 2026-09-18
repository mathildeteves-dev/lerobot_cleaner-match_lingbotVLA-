"""Repeated divided differences at interval midpoints, using source timestamps."""
import numpy as np

from lerobot_cleaner.v21.config import OnMismatch
from lerobot_cleaner.v21.rules.base import CheckRule

from ._common import COLUMNS, stack


class DerivativeRule(CheckRule):
    order = 1

    def check(self, work, context=None):
        metrics = {"targets": {}, "evaluated": False}
        problems = []
        if len(work.df) <= self.order:
            self.stats["episodes_too_short"] += 1
            metrics.update(frames=len(work.df), required_frames=self.order + 1)
            return self.result(work, False, metrics, "insufficient samples for derivative")
        if "timestamp" in work.df:
            times = work.df["timestamp"].to_numpy(dtype=np.float64)
        elif self.fps and np.isfinite(self.fps) and self.fps > 0:
            times = np.asarray(work.keep_indices, dtype=np.float64) / self.fps
        else:
            message = "missing valid timestamps/fps"
            self._problem(work, message)
            return self.result(work, False, metrics, message)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            message = "non-finite or non-increasing timestamps"
            self._problem(work, message)
            return self.result(work, False, metrics, message)
        for col, modality in COLUMNS:
            arr = stack(work, col)
            for dotted, limit in self.config.limits.items():
                if dotted.split(".", 1)[0] != modality:
                    continue
                target = {"threshold": limit, "evaluated": False}
                metrics["targets"][dotted] = target
                if arr is None:
                    problems.append(f"missing data for {dotted}")
                    continue
                sl = self.resolver.resolve(dotted)
                values = arr[:, sl.start:sl.end]
                if not values.size or not np.isfinite(values).all():
                    problems.append(f"missing or non-finite values in {dotted}")
                    continue
                t = times.copy()
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for _ in range(self.order):
                        values = np.diff(values, axis=0) / np.diff(t)[:, None]
                        t = (t[1:] + t[:-1]) / 2
                if not np.isfinite(values).all():
                    problems.append(f"non-finite derivative in {dotted}")
                    continue
                n = int((np.abs(values) > limit).any(axis=1).sum())
                target.update({f"max_{self.name}": float(np.max(np.abs(values))),
                               "intervals_exceeding_limit": n, "evaluated": True})
                if n:
                    self.stats["intervals_exceeding_limit"] += n
                    problems.append(f"{dotted}: {n} derivative intervals exceed {limit}")
        for dotted in self.config.limits:
            if dotted not in metrics["targets"]:
                problems.append(f"unsupported target {dotted}")
        metrics["evaluated"] = bool(metrics["targets"]) and all(
            target["evaluated"] for target in metrics["targets"].values()
        )
        # Single-target convenience fields; multiple targets retain their own units/limits.
        if len(metrics["targets"]) == 1:
            metrics.update(next(iter(metrics["targets"].values())))
        if not metrics["targets"] and not problems:
            problems.append("no configured targets")
        message = "; ".join(problems) or None
        if message:
            self._problem(work, message)
        return self.result(work, not problems, metrics, message)

    def _problem(self, work, message):
        work.note(self.name, message)
        if self.config.on_violation == OnMismatch.strict_drop:
            work.drop(f"{self.name}: {message}")
            self.stats["episodes_dropped"] += 1
