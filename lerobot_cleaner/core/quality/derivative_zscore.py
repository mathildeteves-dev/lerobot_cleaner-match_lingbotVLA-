"""Within-episode, component-wise population z-scores of derivatives."""
import numpy as np

from ._common import positive_threshold, result
from ._derivative import derivative_values


def check_derivative_zscore(trajectory, order, threshold=3.0, source="state", columns=None):
    if order not in (1, 2):
        raise ValueError("derivative z-score supports velocity or acceleration")
    positive_threshold(threshold)
    if threshold is None:
        raise ValueError("z-score threshold is required")
    name = "velocity_zscore" if order == 1 else "acceleration_zscore"
    arr, metrics, message = derivative_values(trajectory, order, source, columns)
    metrics["threshold"] = threshold
    if arr is None:
        return result(name, False, metrics, message)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    # Treat floating-point residue in constant derivatives as zero variation.
    tolerance = np.finfo(float).eps * 64 * np.maximum(1.0, np.max(np.abs(arr), axis=0))
    scores = np.divide(np.abs(arr - mean), std, out=np.zeros_like(arr), where=std > tolerance)
    if not np.isfinite(scores).all() or not np.isfinite(std).all():
        return result(name, False, {**metrics, "evaluated": False}, "non-finite z-score computation")
    n = int((scores > threshold).any(axis=1).sum())
    metrics.update(max_zscore=float(scores.max()), intervals_exceeding_limit=n,
                   reference="within_episode_per_component", ddof=0)
    return result(name, n == 0, metrics, f"{n} derivative z-score outliers" if n else None)


def check_velocity_zscore(trajectory, threshold=3.0, source="state", columns=None):
    return check_derivative_zscore(trajectory, 1, threshold, source, columns)


def check_acceleration_zscore(trajectory, threshold=3.0, source="state", columns=None):
    return check_derivative_zscore(trajectory, 2, threshold, source, columns)
