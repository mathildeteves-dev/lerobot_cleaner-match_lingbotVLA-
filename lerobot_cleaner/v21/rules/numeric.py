"""Expand the legacy numeric_sanity configuration into separate stages."""
from lerobot_cleaner.v21.config import OnBad, OutlierMode
from lerobot_cleaner.v21.rules.base import CheckRule, TransformRule
from lerobot_cleaner.v21.rules.checks.trajectory.acceleration import AccelerationRule
from lerobot_cleaner.v21.rules.checks.trajectory.derivative_zscore import (
    AccelerationZScoreRule,
    VelocityZScoreRule,
)
from lerobot_cleaner.v21.rules.checks.trajectory.finite import FiniteRule
from lerobot_cleaner.v21.rules.checks.trajectory.jerk import JerkRule
from lerobot_cleaner.v21.rules.checks.trajectory.joint_limits import JointLimitsRule
from lerobot_cleaner.v21.rules.checks.trajectory.outlier import OutlierRule
from lerobot_cleaner.v21.rules.checks.trajectory.velocity import VelocityRule
from lerobot_cleaner.v21.rules.transforms.numeric.percentile_clip import PercentileClipRule
from lerobot_cleaner.v21.rules.transforms.numeric.repair import NumericRepairRule


def numeric_checks(config, dataset, bounds=None) -> list[CheckRule]:
    checks = [FiniteRule(config, dataset)]
    if config.joint_limits:
        checks.append(JointLimitsRule(config, dataset))
    for cls in (VelocityRule, AccelerationRule, JerkRule, VelocityZScoreRule, AccelerationZScoreRule):
        cfg = getattr(config, cls.name)
        if cfg.enabled:
            checks.append(cls(cfg, dataset))
    if config.outlier_mode != OutlierMode.off:
        checks.append(OutlierRule(config, dataset, bounds))
    return checks


def numeric_transforms(config, dataset, bounds=None) -> list[TransformRule]:
    transforms = []
    repairs = (OnBad.drop_frame, OnBad.interpolate, OnBad.clip)
    if (config.on_nan in repairs or config.on_inf in repairs
            or (config.joint_limits and config.on_limit_violation in repairs)
            or config.outlier_mode == OutlierMode.drop_frame):
        transforms.append(NumericRepairRule(config, dataset, bounds))
    if config.outlier_mode == OutlierMode.clip_quantile:
        transforms.append(PercentileClipRule(config, dataset, bounds))
    return transforms
