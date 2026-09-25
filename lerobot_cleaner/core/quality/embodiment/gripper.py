"""Check an explicitly selected gripper convention; never infer or binarize it."""
import numpy as np
from .._common import result, values


def check_gripper(view, source, columns, mode="continuous", low=0., high=1., tolerance=1e-6):
    arr = values(view, source, columns)
    if not arr.size or not np.isfinite(arr).all():
        return result("gripper", False, {"evaluated": False}, "missing/nonfinite gripper values")
    valid = (arr >= low - tolerance) & (arr <= high + tolerance)
    if mode != "continuous":
        allowed = [0., 1.] if mode == "binary_01" else [-1., 1.]
        valid = np.any(np.isclose(arr[..., None], allowed, atol=tolerance, rtol=0), axis=-1)
    bad = int((~valid).any(axis=1).sum())
    return result("gripper", bad == 0, {"evaluated": True, "source": source,
        "columns": columns, "mode": mode, "violating_frames": bad},
        None if not bad else "gripper values violate the configured convention")
