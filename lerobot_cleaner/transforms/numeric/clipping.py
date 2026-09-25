"""Clipping uses explicit physical/reference bounds; it never fits on a shard."""
import numpy as np


def clip_values(values, low, high):
    low, high = np.asarray(low), np.asarray(high)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ValueError("Clipping bounds must be finite and ordered")
    return np.clip(values, low, high)
