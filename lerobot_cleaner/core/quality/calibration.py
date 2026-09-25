"""Dataset-independent robust calibration of nonnegative quality measurements."""
import numpy as np


def robust_threshold(samples, *, quantile=0.995, mad_k=8.0, mad_scale=1.4826,
                     min_samples=20, positive_floor=1e-12):
    """Exact linear quantile plus scaled MAD; input contains one peak per episode.

    Nonfinite/negative samples are excluded explicitly. The positive floor only
    handles zero distributions, since quality-check thresholds must be positive.
    """
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if not np.isfinite(mad_k) or mad_k < 0:
        raise ValueError("mad_k must be finite and nonnegative")
    if not np.isfinite(mad_scale) or mad_scale <= 0:
        raise ValueError("mad_scale must be finite and positive")
    if type(min_samples) is not int or min_samples < 2:
        raise ValueError("min_samples must be an integer >= 2")
    if not np.isfinite(positive_floor) or positive_floor <= 0:
        raise ValueError("positive_floor must be finite and positive")
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1:
        raise ValueError("samples must be a 1D array")
    valid = values[np.isfinite(values) & (values >= 0)]
    report = {"count": len(valid), "excluded": len(values) - len(valid),
              "status": "insufficient_samples", "threshold": None}
    if len(valid) < min_samples:
        return report
    with np.errstate(over="ignore", invalid="ignore"):
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        tq = float(np.quantile(valid, quantile, method="linear"))
        tmad = float(median + mad_k * mad_scale * mad)
    if not np.isfinite([median, mad, tq, tmad]).all():
        report["status"] = "numeric_overflow"
        return report
    raw = max(tq, tmad)
    threshold = max(raw, positive_floor)
    report.update(status="ready", median=median, mad=mad, scaled_mad=mad_scale * mad,
                  quantile_threshold=tq, mad_threshold=tmad, raw_threshold=raw,
                  threshold=threshold, floor_applied=raw < positive_floor,
                  samples_exceeding_threshold=int((valid > threshold).sum()),
                  expected_upper_tail_samples=float(len(valid) * (1 - quantile)))
    return report
