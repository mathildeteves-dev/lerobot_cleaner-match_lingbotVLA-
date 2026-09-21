"""Execute numeric edits on a working copy, using source timestamps for repairs."""
import numpy as np


def apply_numeric(episode, edit, info):
    if edit.column in {"index", "episode_index", "frame_index", "timestamp", "task_index"}:
        raise ValueError("Numeric edits cannot target identity columns")
    spec = info["features"][edit.column]
    from lerobot_cleaner.adapters.episode_access import sensor_values
    scalar = np.isscalar(episode.df[edit.column].iloc[0])
    shape = (len(episode.df),) if scalar and int(np.prod(spec["shape"])) == 1 else (len(episode.df), *spec["shape"])
    arr = sensor_values(episode.df[edit.column], spec["shape"]).astype(float)
    if not 0 <= edit.start < edit.end <= arr.shape[1]:
        raise ValueError("Numeric edit outside source dimensions")
    values = arr[:, edit.start:edit.end].copy()
    if edit.operation == "interpolate":
        times = np.asarray(episode.metadata["source_timestamps"], dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Interpolation requires finite, increasing source timestamps")
        for dim in range(values.shape[1]):
            valid = np.isfinite(values[:, dim])
            if not valid.any():
                raise ValueError(f"Cannot interpolate entirely missing {edit.column}")
            values[~valid, dim] = np.interp(times[~valid], times[valid], values[valid, dim])
    elif edit.operation in {"clip", "percentile_clip"}:
        from .clipping import clip_values
        values = clip_values(values, edit.parameters["low"], edit.parameters["high"])
    elif edit.operation == "gripper":
        from ..embodiment.gripper import binarize
        values = binarize(values, edit.parameters)
    elif edit.operation == "alias":
        source = np.stack(episode.df[edit.parameters["source"]].to_numpy()).reshape(len(arr), -1)
        values = source[:, edit.parameters["start"]:edit.parameters["end"]]
    else:
        raise ValueError(f"Unsupported numeric edit: {edit.operation}")
    changed = int((~np.isclose(values, arr[:, edit.start:edit.end], rtol=0, atol=0, equal_nan=True)).sum())
    arr[:, edit.start:edit.end] = values
    result = arr.astype(spec["dtype"]).reshape(shape)
    episode.df[edit.column] = result if result.ndim == 1 else list(result)
    return changed
