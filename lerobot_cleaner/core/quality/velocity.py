"""Version-independent velocity check."""
from ._derivative import check_derivative


def check_velocity(trajectory, threshold=None, source="state", columns=None):
    return check_derivative(trajectory, 1, "velocity", threshold, source, columns)
