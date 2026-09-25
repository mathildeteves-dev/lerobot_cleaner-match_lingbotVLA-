"""Pure visual metrics on normalized HWC pixels; no storage or file access."""
import numpy as np
from .._common import result


def pixel_metrics(pixels, black_level=None):
    gray = np.asarray(pixels, dtype=float)[..., :3].mean(axis=-1)
    return {"brightness": float(gray.mean()),
            "black": bool(np.mean(gray <= black_level) >= .98) if black_level is not None else False}


def check_black(samples, max_ratio, complete=True):
    ratio = float(np.mean([s["black"] for s in samples])) if samples else None
    evaluated = bool(samples) and complete
    return result("black_frame", evaluated and ratio <= max_ratio,
                  {"evaluated": evaluated, "black_ratio": ratio, "samples": len(samples), "max_ratio": max_ratio})


def check_brightness(samples, low=None, high=None, complete=True):
    values = [s["brightness"] for s in samples]
    evaluated = bool(values) and complete
    passed = evaluated and all((low is None or v >= low) and (high is None or v <= high) for v in values)
    return result("brightness", passed, {"evaluated": evaluated, "samples": len(values),
        "min": min(values) if values else None, "max": max(values) if values else None,
        "low": low, "high": high})
