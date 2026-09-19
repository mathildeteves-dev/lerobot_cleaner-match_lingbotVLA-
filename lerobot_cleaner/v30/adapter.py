"""Convert a version-specific episode into the core array contract."""
import numpy as np

from lerobot_cleaner.core.trajectory import TrajectoryView


def _matrix(frame, key):
    if key not in frame:
        return np.empty((len(frame), 0))
    if not len(frame):
        return np.empty((0, 0))
    arr = np.stack(frame[key].to_numpy()).astype(float)
    return arr[:, None] if arr.ndim == 1 else arr


class V30Adapter:
    @staticmethod
    def to_trajectory(frame, fps, state_column="observation.state", action_column="action"):
        """Frame must be a single, ordered episode slice supplied by the v3 reader."""
        if "episode_index" in frame and frame["episode_index"].nunique() > 1:
            raise ValueError("V30Adapter requires a single episode slice")
        if "timestamp" in frame:
            timestamps = frame["timestamp"].to_numpy(dtype=float)
        elif "frame_index" in frame:
            timestamps = frame["frame_index"].to_numpy(dtype=float) / fps
        else:
            timestamps = np.arange(len(frame), dtype=float) / fps
        return TrajectoryView(_matrix(frame, state_column), _matrix(frame, action_column), timestamps, fps)
