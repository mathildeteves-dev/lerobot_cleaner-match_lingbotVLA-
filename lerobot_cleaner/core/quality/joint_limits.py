"""Detect finite values outside explicit bounds in native target units."""
import numpy as np

from ._common import result, values


def check_joint_limits(trajectory, low, high, source="state", columns=None):
    arr = values(trajectory, source, columns)
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError("joint bounds must be finite and ordered")
    if not arr.size:
        return result("joint_limits", False, {"evaluated": False}, "missing numeric data")
    mask = np.isfinite(arr) & ((arr < low) | (arr > high))
    n = int(mask.any(axis=1).sum())
    return result("joint_limits", n == 0, {"evaluated": True, "violating_frames": n,
        "low": low.tolist(), "high": high.tolist()}, f"{n} joint limit violations" if n else None)
