"""Version-independent jerk check."""
from ._derivative import check_derivative


def check_jerk(trajectory, threshold=None, source="state", columns=None):
    return check_derivative(trajectory, 3, "jerk", threshold, source, columns)
