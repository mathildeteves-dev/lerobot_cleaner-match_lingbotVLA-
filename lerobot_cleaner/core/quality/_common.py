"""Shared pure array selection and result construction."""
import numpy as np

from lerobot_cleaner.core.trajectory import CheckResult


def values(trajectory, source="state", columns=None):
    if source not in ("state", "action"):
        raise ValueError("source must be state or action")
    arr = getattr(trajectory, source)
    if columns is not None:
        arr = arr[:, columns]
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def result(rule, passed, metrics, message=None):
    return CheckResult(bool(passed), rule, "info" if passed else "warning", metrics, message)


def positive_threshold(threshold):
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("threshold must be finite and positive")
