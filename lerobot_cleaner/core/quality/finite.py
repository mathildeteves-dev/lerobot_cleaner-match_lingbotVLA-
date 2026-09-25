"""Detect non-finite trajectory values without altering inputs."""
import numpy as np

from ._common import result


def check_finite(trajectory):
    metrics = {}
    for name in ("state", "action"):
        arr = getattr(trajectory, name)
        metrics[name] = {"nan_frames": int(np.isnan(arr).any(axis=1).sum()),
                         "inf_frames": int(np.isinf(arr).any(axis=1).sum())}
    metrics["timestamps"] = {"nonfinite_values": int((~np.isfinite(trajectory.timestamps)).sum())}
    passed = not any(count for group in metrics.values() for count in group.values())
    return result("finite", passed, metrics, None if passed else "non-finite trajectory values")
