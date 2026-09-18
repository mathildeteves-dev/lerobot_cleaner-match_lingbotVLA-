"""Read-only jerk magnitude check on explicitly configured numeric targets."""
from ._derivative import DerivativeRule


class JerkRule(DerivativeRule):
    name = "jerk"
    order = 3
