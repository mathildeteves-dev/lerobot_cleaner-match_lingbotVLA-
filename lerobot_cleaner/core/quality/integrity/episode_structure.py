"""Check row identity against official episode metadata supplied by the caller."""
import numpy as np
from .._common import result


def check_episode_structure(columns, length, episode_index, metadata=None):
    metadata = metadata or {}
    errors = []
    for name in ("episode_index", "frame_index", "index", "timestamp", "task_index"):
        if name not in columns:
            errors.append(f"missing {name}")
    if length == 0:
        errors.append("empty episode")
    if "episode_index" in columns and not np.all(np.asarray(columns["episode_index"]) == episode_index):
        errors.append("mixed episode indices")
    if "frame_index" in columns and not np.array_equal(columns["frame_index"], np.arange(length)):
        errors.append("noncontiguous frame indices")
    if "index" in columns:
        indices = np.asarray(columns["index"])
        if len(indices) > 1 and not np.all(np.diff(indices) == 1):
            errors.append("noncontiguous global indices")
        start = metadata.get("dataset_from_index")
        if start is not None and not np.array_equal(indices, np.arange(start, start + length)):
            errors.append("official start offset mismatch")
    if metadata.get("length", length) != length:
        errors.append("official length mismatch")
    if "dataset_to_index" in metadata and "dataset_from_index" in metadata:
        if metadata["dataset_to_index"] - metadata["dataset_from_index"] != length:
            errors.append("official interval mismatch")
    return result("episode_structure", not errors, {"evaluated": True, "frames": length,
        "metadata_available": bool(metadata), "errors": errors}, "; ".join(errors) or None)


def check_episode_length(view, min_frames=1, max_frames=None, min_seconds=None, max_seconds=None):
    count = len(view.timestamps)
    duration = count / view.fps
    passed = (count >= min_frames and (max_frames is None or count <= max_frames)
              and (min_seconds is None or duration >= min_seconds)
              and (max_seconds is None or duration <= max_seconds))
    return result("episode_length", passed, {"evaluated": True, "frames": count,
        "duration_seconds": duration, "min_frames": min_frames, "max_frames": max_frames,
        "min_seconds": min_seconds, "max_seconds": max_seconds},
        None if passed else "episode length outside configured limits")
