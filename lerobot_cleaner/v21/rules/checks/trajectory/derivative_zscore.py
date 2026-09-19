"""v2.1 policies for shared derivative z-score computations."""
from ._derivative import DerivativeRule


class VelocityZScoreRule(DerivativeRule):
    name = "velocity_zscore"
    order = 1


class AccelerationZScoreRule(DerivativeRule):
    name = "acceleration_zscore"
    order = 2
