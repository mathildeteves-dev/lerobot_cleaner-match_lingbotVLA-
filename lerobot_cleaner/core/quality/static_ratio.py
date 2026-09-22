"""Fraction of consecutive transitions below a native-unit step tolerance."""
import numpy as np
from lerobot_cleaner.core.physical import PhysicalDeltaResolver

from ._common import result, values


def check_static_ratio(trajectory, epsilon=1e-4, threshold=None, source="state", columns=None):
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("static ratio threshold must be within [0, 1]")
    arr = values(trajectory, source, columns)
    metrics = {"evaluated": False, "epsilon": epsilon, "threshold": threshold}
    if len(arr) < 2 or not arr.size or not np.isfinite(arr).all():
        return result("static_ratio", False, metrics, "insufficient or non-finite trajectory")
    try:
        delta = PhysicalDeltaResolver.trajectory_difference(trajectory, source, columns)
    except ValueError as exc:
        return result("static_ratio", False, {"evaluated": False}, str(exc))
    count = int((np.abs(delta).max(axis=1) <= epsilon).sum())
    ratio = count / (len(arr) - 1)
    metrics.update(evaluated=True, static_ratio=ratio, static_transitions=count,
                   transitions=len(arr)-1, threshold_applied=threshold is not None)
    passed = threshold is None or ratio <= threshold
    return result("static_ratio", passed, metrics, None if passed else "static ratio exceeds threshold")
