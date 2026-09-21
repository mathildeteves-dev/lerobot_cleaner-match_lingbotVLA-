"""Read-only Laplacian variance heuristic, independent of storage/decoders."""
import numpy as np
from .._common import result


def blur_variance(rgb):
    image = np.asarray(rgb, dtype=float)
    gray = image[..., :3].mean(axis=-1)
    if min(gray.shape) < 3 or not np.isfinite(gray).all():
        return None
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(lap.var())


def check_blur(variances, threshold, maximum_ratio):
    valid = [value for value in variances if value is not None]
    if not valid or len(valid) != len(variances):
        return result("blur", False, {"evaluated": False}, "missing/invalid decoded images")
    ratio = sum(value < threshold for value in valid) / len(valid)
    return result("blur", ratio <= maximum_ratio, {"evaluated": True, "samples": len(valid),
        "blur_ratio": ratio, "variance_threshold": threshold, "maximum_ratio": maximum_ratio,
        "heuristic": True}, "low image detail is not proof of defocus" if ratio > maximum_ratio else None)
