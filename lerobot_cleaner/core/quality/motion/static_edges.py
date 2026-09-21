"""Read-only edge-trim proposal; does not mutate identity or timestamps."""
import numpy as np
from .._common import result, values


def check_static_edges(view, source="action", columns=None, epsilon=.002, min_kept_frames=2):
    arr = values(view, source, columns)
    if not arr.size or not np.isfinite(arr).all():
        return result("static_edges", False, {"evaluated": False}, "missing/nonfinite motion values")
    motion = np.max(np.abs(np.diff(arr, axis=0)), axis=1)
    moving = np.flatnonzero(motion >= epsilon)
    drop_frames = (np.flatnonzero(motion < epsilon) + 1).tolist()
    if not len(moving):
        return result("static_edges", False, {"evaluated": True, "all_static": True, "static_ratio": 1., "should_trim": False,
            "trim_start": 0, "trim_end": len(arr),
            "proposal": None, "drop_frames": drop_frames, "applied": False}, "no motion; no automatic destructive trim")
    start, stop = int(moving[0]), int(moving[-1]) + 2
    can_trim = stop - start >= min_kept_frames and (start > 0 or stop < len(arr))
    return result("static_edges", not can_trim, {"evaluated": True, "all_static": False,
        "static_ratio": len(drop_frames) / max(1, len(arr)-1), "should_trim": can_trim,
        "trim_start": start, "trim_end": stop,
        "leading_frames": start, "trailing_frames": len(arr)-stop,
        "proposal": [start, stop] if can_trim else None, "drop_frames": drop_frames, "applied": False},
        "static edges can be reviewed for trimming" if can_trim else None)
