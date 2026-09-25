"""Measure timestamp quality without relabeling or resampling time."""
import numpy as np
from .._common import result


def check_timestamp(view, tolerance_ratio=.2, require_uniform=True, require_zero_start=True):
    ts = view.timestamps
    finite = bool(np.isfinite(ts).all())
    dt = np.diff(ts)
    monotonic = finite and bool((dt > 0).all())
    expected = 1 / view.fps
    mean_dt = float(dt.mean()) if len(dt) and finite else None
    uniform = finite and bool(np.all(np.abs(dt - expected) <= expected * tolerance_ratio))
    mean_ok = mean_dt is None or abs(mean_dt - expected) <= expected * tolerance_ratio
    start_ok = not require_zero_start or (len(ts) > 0 and abs(ts[0]) <= expected * tolerance_ratio)
    passed = bool(len(ts)) and finite and monotonic and mean_ok and start_ok and (uniform or not require_uniform)
    return result("timestamp", passed, {"evaluated": True, "frames": len(ts),
        "finite": finite, "strictly_increasing": monotonic, "uniform": uniform,
        "zero_start": start_ok, "expected_dt": expected, "mean_dt": mean_dt},
        None if passed else "timestamp cadence/alignment violation")
