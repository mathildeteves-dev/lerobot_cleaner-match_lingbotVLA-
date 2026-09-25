"""Explicit gripper mutation, separate from gripper convention checks."""
import numpy as np


def binarize(values, settings):
    if not np.isfinite(values).all():
        raise ValueError("Repair nonfinite gripper values before binarization")
    low = 0. if settings.get("mode", "binary_01") == "binary_01" else -1.
    return np.where(values >= settings.get("threshold", .5), 1., low)
