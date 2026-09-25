"""Canonical episode identity and numeric sensor access, without mutations."""
import hashlib
import json
import numpy as np

IDENTITY_COLUMNS = {"index", "frame_index", "episode_index", "timestamp", "task_index"}


def sensor_values(series, shape):
    """Flatten only for analysis; output writers keep original feature shapes."""
    width = int(np.prod(shape))
    rows = []
    for value in series:
        array = np.asarray(value)
        if value is None or (array.ndim == 0 and np.issubdtype(array.dtype, np.number) and np.isnan(array)):
            rows.append(np.full(width, np.nan))
        else:
            if array.dtype == object:
                array = np.asarray(value, dtype=float)
            if array.size != width:
                raise ValueError("Sensor cell does not match declared feature shape")
            rows.append(array.reshape(width))
    return np.stack(rows) if rows else np.empty((0, width))


def episode_identity(episode):
    rows = {key: episode.df[key].tolist() for key in sorted(IDENTITY_COLUMNS) if key in episode.df}
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


