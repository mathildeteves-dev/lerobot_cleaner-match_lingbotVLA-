"""Per-dimension population z-score within one trajectory; no clipping."""
import numpy as np

from ._common import positive_threshold, result, values


def check_zscore(trajectory, threshold=3.0, source="state", columns=None):
    positive_threshold(threshold)
    if threshold is None:
        raise ValueError("z-score requires an explicit threshold")
    arr = values(trajectory, source, columns)
    if not arr.size or not np.isfinite(arr).all():
        return result("zscore", False, {"evaluated": False}, "missing or non-finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        std = arr.std(axis=0)
        scores = np.divide(np.abs(arr - arr.mean(axis=0)), std, out=np.zeros_like(arr), where=std > 0)
    if not np.isfinite(scores).all() or not np.isfinite(std).all():
        return result("zscore", False, {"evaluated": False}, "non-finite z-score computation")
    n = int((scores > threshold).any(axis=1).sum())
    return result("zscore", n == 0, {"evaluated": True, "max_zscore": float(scores.max()),
        "threshold": threshold, "outlier_frames": n}, f"{n} z-score outlier frames" if n else None)
