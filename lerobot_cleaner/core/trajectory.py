"""Pure array contract; no dataset format, storage, or policy dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


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


@dataclass(frozen=True)
class TrajectoryView:
    state: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray
    fps: float

    def __post_init__(self):
        state = np.array(self.state, dtype=float, copy=True)
        action = np.array(self.action, dtype=float, copy=True)
        timestamps = np.array(self.timestamps, dtype=float, copy=True)
        if state.ndim != 2 or action.ndim != 2:
            raise ValueError("state/action must be 2D arrays [frames, dimensions]")
        if timestamps.ndim != 1 or len(state) != len(action) or len(state) != len(timestamps):
            raise ValueError("state, action, and timestamps must have equal frame counts")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        for name, value in (("state", state), ("action", action), ("timestamps", timestamps)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
