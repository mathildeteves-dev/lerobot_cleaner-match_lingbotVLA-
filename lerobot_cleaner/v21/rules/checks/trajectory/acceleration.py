"""Read-only acceleration magnitude check on explicitly configured numeric targets."""
from ._derivative import DerivativeRule


class AccelerationRule(DerivativeRule):
    name = "acceleration"
    order = 2
