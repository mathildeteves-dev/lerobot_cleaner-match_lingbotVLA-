"""Read-only numeric array helpers shared with numeric transforms."""
import numpy as np

from lerobot_cleaner.v21.reader import ACTION_COL, STATE_COL

COLUMNS = ((STATE_COL, "state"), (ACTION_COL, "action"))


def stack(work, col):
    if col not in work.df or not len(work.df):
        return None
    return np.stack(work.df[col].to_numpy()).astype(np.float64)


def target_mask(arr, modality, targets, resolver):
    mask = np.zeros(arr.shape[1], dtype=bool) if targets else np.ones(arr.shape[1], dtype=bool)
    for dotted in targets:
        if dotted.split(".", 1)[0] == modality:
            sl = resolver.resolve(dotted)
            mask[sl.start:sl.end] = True
    return mask


def outlier_mask(arr, low, high, targets):
    return np.isfinite(arr) & ((arr < low) | (arr > high)) & targets


def finite_quantile_bounds(values, low_quantile, high_quantile):
    """Ignore non-finite samples; unconstrained bounds for dimensions with no data."""
    low = np.full(values.shape[1], -np.inf)
    high = np.full(values.shape[1], np.inf)
    for j in range(values.shape[1]):
        finite = values[:, j][np.isfinite(values[:, j])]
        if len(finite):
            low[j], high[j] = np.quantile(finite, [low_quantile, high_quantile])
    return low, high
