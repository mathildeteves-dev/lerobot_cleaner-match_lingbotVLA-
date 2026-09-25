"""Check configured joint ranges without clipping or removing frames."""
from lerobot_cleaner.core.quality import check_joint_limits
from lerobot_cleaner.v21.adapter import V21Adapter
from lerobot_cleaner.v21.config import OnBad
from lerobot_cleaner.v21.rules.base import CheckRule

from ._common import COLUMNS, stack


class JointLimitsRule(CheckRule):
    name = "joint_limits"

    def check(self, work, context=None):
        metrics = {}
        problems = []
        trajectory = V21Adapter.to_trajectory(work, self.fps)
        for col, modality in COLUMNS:
            arr = stack(work, col)
            if arr is None:
                continue
            for dotted, (low, high) in self.config.joint_limits.items():
                if dotted.split(".", 1)[0] != modality:
                    continue
                source, columns = V21Adapter.resolve_target(self.resolver, dotted)
                measured = check_joint_limits(trajectory, low, high, source=source, columns=columns)
                n = measured.metrics.get("violating_frames", 0)
                metrics[dotted] = {"violating_frames": n, "low": low, "high": high}
                if not n:
                    continue
                problems.append(f"{dotted}: {n} joint limit violations")
                self.stats["limit_violations"] += n
                work.note(f"{self.name}:{dotted}", f"{n} joint limit violations")
                if self.config.on_limit_violation == OnBad.drop_episode:
                    work.drop(f"joint limit violation in {dotted}")
                    self.stats["episodes_dropped_limits"] += 1
                    return self.result(work, False, metrics, "; ".join(problems))

        return self.result(work, not problems, metrics, "; ".join(problems) or None)
