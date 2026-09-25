"""Repeated divided differences at interval midpoints, using source timestamps."""
from lerobot_cleaner.core.quality import (
    check_acceleration,
    check_acceleration_zscore,
    check_jerk,
    check_velocity,
    check_velocity_zscore,
)
from lerobot_cleaner.v21.adapter import V21Adapter
from lerobot_cleaner.v21.config import OnMismatch
from lerobot_cleaner.v21.rules.base import CheckRule


class DerivativeRule(CheckRule):
    order = 1

    def check(self, work, context=None):
        trajectory = V21Adapter.to_trajectory(work, self.fps)
        metrics = {"targets": {}, "evaluated": False}
        problems = []
        if len(work.df) <= self.order:
            self.stats["episodes_too_short"] += 1
            metrics.update(frames=len(work.df), required_frames=self.order + 1)
            return self.result(work, False, metrics, "insufficient samples for derivative")
        function = {1: check_velocity, 2: check_acceleration, 3: check_jerk}[self.order]
        function = {"velocity_zscore": check_velocity_zscore, "acceleration_zscore": check_acceleration_zscore}.get(self.name, function)
        for dotted, limit in self.config.limits.items():
            source, columns = V21Adapter.resolve_target(self.resolver, dotted)
            measured = function(trajectory, threshold=limit, source=source, columns=columns)
            metrics["targets"][dotted] = measured.metrics
            self.stats["intervals_exceeding_limit"] += measured.metrics.get("intervals_exceeding_limit", 0)
            if not measured.passed:
                problems.append(f"{dotted}: {measured.message}")
        metrics["evaluated"] = bool(metrics["targets"]) and all(
            target["evaluated"] for target in metrics["targets"].values()
        )
        if len(metrics["targets"]) == 1:
            metrics.update(next(iter(metrics["targets"].values())))
        if not metrics["targets"]:
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
