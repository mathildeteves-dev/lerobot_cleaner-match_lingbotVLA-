"""Outliers against explicit reference bounds; never fit thresholds on a shard."""
from .joint_limits import check_joint_limits


def check_percentile(view, low, high, source="state", columns=None):
    check = check_joint_limits(view, low, high, source, columns)
    check.rule = "percentile_outlier"
    check.metrics["reference"] = "explicit calibrated percentile bounds"
    return check
