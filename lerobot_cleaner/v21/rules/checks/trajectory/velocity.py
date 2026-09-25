"""Read-only velocity magnitude check on explicitly configured numeric targets."""
from ._derivative import DerivativeRule


class VelocityRule(DerivativeRule):
    name = "velocity"
    order = 1
