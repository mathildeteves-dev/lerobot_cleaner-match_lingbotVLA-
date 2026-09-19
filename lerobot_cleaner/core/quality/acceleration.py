"""Version-independent acceleration check."""
from ._derivative import check_derivative


def check_acceleration(trajectory, threshold=None, source="state", columns=None):
    return check_derivative(trajectory, 2, "acceleration", threshold, source, columns)
