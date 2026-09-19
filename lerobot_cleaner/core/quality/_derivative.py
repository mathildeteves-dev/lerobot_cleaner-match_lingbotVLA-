"""Repeated divided differences with midpoint timestamps, in seconds."""
import numpy as np

from ._common import positive_threshold, result, values


def derivative_values(trajectory, order, source="state", columns=None):
    arr = values(trajectory, source, columns)
    metrics = {"evaluated": False}
    if len(arr) <= order:
        metrics.update(frames=len(arr), required_frames=order + 1)
        return None, metrics, "insufficient samples for derivative"
    t = trajectory.timestamps
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return None, metrics, "non-finite or non-increasing timestamps"
    if not arr.size or not np.isfinite(arr).all():
        return None, metrics, "missing or non-finite values"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(order):
            arr = np.diff(arr, axis=0) / np.diff(t)[:, None]
            t = (t[1:] + t[:-1]) / 2
    if not np.isfinite(arr).all():
        return None, metrics, "non-finite derivative"
    return arr, {"evaluated": True}, None


def check_derivative(trajectory, order, name, threshold=None, source="state", columns=None):
    positive_threshold(threshold)
    arr, metrics, message = derivative_values(trajectory, order, source, columns)
    metrics["threshold"] = threshold
    if arr is None:
        return result(name, False, metrics, message)
    count = int((np.abs(arr) > threshold).any(axis=1).sum()) if threshold is not None else 0
    metrics.update({"evaluated": True, f"max_{name}": float(np.max(np.abs(arr))),
                    "intervals_exceeding_limit": count, "threshold_applied": threshold is not None})
    return result(name, count == 0, metrics, f"{count} derivative intervals exceed {threshold}" if count else None)
