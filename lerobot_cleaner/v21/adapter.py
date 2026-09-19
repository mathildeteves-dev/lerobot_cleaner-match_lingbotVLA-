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


class V21Adapter:
    @staticmethod
    def to_trajectory(episode, fps):
        frame = episode.df
        timestamps = frame["timestamp"].to_numpy(dtype=float) if "timestamp" in frame else np.asarray(episode.keep_indices, dtype=float) / fps
        return TrajectoryView(_matrix(frame, "observation.state"), _matrix(frame, "action"), timestamps, fps)

    @staticmethod
    def resolve_target(resolver, dotted):
        resolved = resolver.resolve(dotted)
        return resolved.modality, slice(resolved.start, resolved.end)
