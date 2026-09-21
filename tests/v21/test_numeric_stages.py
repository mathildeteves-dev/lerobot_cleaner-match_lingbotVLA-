"""Numeric checks are read-only; repairs preserve source-frame alignment."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from lerobot_cleaner.v21.config import CleaningConfig, DerivativeCheckConfig, NumericSanityConfig
from lerobot_cleaner.v21.reader import ModalityResolver
from lerobot_cleaner.v21.rules import build_checks, build_transforms, run_episode_stages
from lerobot_cleaner.v21.rules.checks.integrity.numeric import NumericSanityRule
from lerobot_cleaner.v21.rules.checks.trajectory.acceleration import AccelerationRule
from lerobot_cleaner.v21.rules.checks.trajectory.jerk import JerkRule
from lerobot_cleaner.v21.rules.checks.trajectory.velocity import VelocityRule
from lerobot_cleaner.v21.rules.numeric import numeric_checks, numeric_transforms
from lerobot_cleaner.v21.types import EpisodeWork


def dataset():
    return SimpleNamespace(fps=10, resolver=ModalityResolver({
        "state": {"arm": {"start": 0, "end": 1}, "gripper": {"start": 1, "end": 2}},
        "action": {"arm": {"start": 0, "end": 1}, "gripper": {"start": 1, "end": 2}},
    }))


def work(values, times=None):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack((arr, np.zeros(len(arr))))
    df = pd.DataFrame({"observation.state": list(arr.copy()), "action": list(arr.copy()),
                       "timestamp": np.arange(len(arr)) / 10 if times is None else times})
    return EpisodeWork(ref=SimpleNamespace(video_paths={}), df=df, keep_indices=list(range(len(arr))))


BOUNDS = {"state": (np.array([-1., -1.]), np.array([1., 1.]))}


def test_checks_preserve_all_values_and_indices_with_repair_policies():
    cfg = NumericSanityConfig(on_nan="interpolate", on_inf="clip",
        joint_limits={"state.arm": [-2, 2]}, on_limit_violation="clip", outlier_mode="clip_quantile")
    w = work([0, np.nan, np.inf, 10])
    before = np.stack(w.df["observation.state"]).copy()
    for check in numeric_checks(cfg, dataset(), BOUNDS):
        check.apply(w)
    np.testing.assert_equal(np.stack(w.df["observation.state"]), before)
    assert w.keep_indices == [0, 1, 2, 3]
    assert not w.dropped
    assert w.notes


def test_pipeline_expands_legacy_config_and_passes_bounds_to_transform():
    cfg = CleaningConfig(rules={"numeric_sanity": {"enabled": True,
        "outlier_mode": "clip_quantile", "outlier_targets": ["state.arm"]}})
    checks = build_checks(cfg, dataset(), BOUNDS)
    transforms = build_transforms(cfg, dataset(), BOUNDS)
    assert [r.name for r in checks] == ["finite", "outlier"]
    assert [r.name for r in transforms] == ["percentile_clip"]
    w = work([[5, 20], [-5, 30]])
    stats = run_episode_stages(w, checks, transforms)
    np.testing.assert_equal(np.stack(w.df["observation.state"]), [[1, 20], [-1, 30]])
    np.testing.assert_equal(np.stack(w.df["action"]), [[5, 20], [-5, 30]])
    assert stats["outlier"]["outlier_frames_state"] == 2
    assert stats["percentile_clip"]["outlier_frames_clipped_state"] == 2


def test_rejected_episode_is_not_repaired():
    cfg = NumericSanityConfig(on_nan="drop_episode", outlier_mode="clip_quantile")
    w = work([np.nan, 100])
    stats = run_episode_stages(w, numeric_checks(cfg, dataset(), BOUNDS), numeric_transforms(cfg, dataset(), BOUNDS))
    assert w.dropped
    assert "percentile_clip" not in stats
    assert np.stack(w.df["observation.state"])[1, 0] == 100


def test_mixed_nan_inf_deletions_use_current_masks_and_keep_alignment():
    cfg = NumericSanityConfig(on_nan="drop_frame", on_inf="drop_frame", outlier_mode="off")
    w = work([np.nan, 1, np.inf, 2])
    NumericSanityRule(cfg, dataset()).apply(w)
    assert not w.dropped
    assert w.keep_indices == [1, 3]
    np.testing.assert_equal(np.stack(w.df["observation.state"])[:, 0], [1, 2])


def test_all_frames_removed_is_clean_rejection():
    cfg = NumericSanityConfig(on_nan="drop_frame", outlier_mode="off")
    w = work([np.nan, np.nan])
    NumericSanityRule(cfg, dataset()).apply(w)
    assert w.dropped and not len(w.df) and w.keep_indices == []


def test_repairs_compose_without_losing_earlier_joint_clips():
    cfg = NumericSanityConfig(joint_limits={"state.arm": [-1, 1]}, outlier_mode="drop_frame",
                             outlier_targets=["state.gripper"])
    w = work([[5, 0], [5, 100], [5, 0]])
    NumericSanityRule(cfg, dataset(), BOUNDS).apply(w)
    assert w.keep_indices == [0, 2]
    np.testing.assert_equal(np.stack(w.df["observation.state"])[:, 0], [1, 1])


@pytest.mark.parametrize("cls,order,value", [(VelocityRule, 1, 2), (AccelerationRule, 2, 2), (JerkRule, 3, 6)])
def test_derivatives_use_seconds_and_are_read_only(cls, order, value):
    t = np.arange(6, dtype=float) * 0.2
    values = 2*t if order == 1 else t**order
    w = work(values, t)
    before = np.stack(w.df["observation.state"]).copy()
    cls(DerivativeCheckConfig(enabled=True, limits={"state.arm": value * 1.1}), dataset()).apply(w)
    assert not w.notes
    rule = cls(DerivativeCheckConfig(enabled=True, limits={"state.arm": value * .9}, on_violation="strict_drop"), dataset())
    rule.apply(w)
    assert w.dropped
    np.testing.assert_equal(np.stack(w.df["observation.state"]), before)


def test_velocity_uses_irregular_timestamps():
    t = np.array([0., .1, .4, 1.])
    w = work(t * 2, t)
    VelocityRule(DerivativeCheckConfig(enabled=True, limits={"state.arm": 2.1}), dataset()).apply(w)
    assert not w.notes


@pytest.mark.parametrize("times", [[0, 0, 1], [0, np.nan, 1]])
def test_invalid_times_are_reported_without_dividing(times):
    w = work([0, 1, 2], times)
    VelocityRule(DerivativeCheckConfig(enabled=True, limits={"state.arm": 10}, on_violation="strict_drop"), dataset()).apply(w)
    assert w.dropped


def test_derivative_configuration_is_opt_in_and_validated():
    cfg = NumericSanityConfig()
    assert not cfg.velocity.enabled and not cfg.acceleration.enabled and not cfg.jerk.enabled
    with pytest.raises(ValidationError):
        DerivativeCheckConfig(enabled=True)
    with pytest.raises(ValidationError):
        DerivativeCheckConfig(limits={"state.arm": -1})


def test_quantile_prepass_ignores_nonfinite_samples():
    from lerobot_cleaner.v21.pipeline import Pipeline
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = CleaningConfig(rules={"numeric_sanity": {"enabled": True, "outlier_mode": "clip_quantile"}})
    w = work([[0, np.nan], [2, np.inf], [np.nan, -np.inf]])
    pipeline.source = SimpleNamespace(read_episode=lambda index: w)
    bounds = pipeline._compute_outlier_bounds([SimpleNamespace(episode_index=0)])
    low, high = bounds["state"]
    np.testing.assert_allclose([low[0], high[0]], [.02, 1.98])
    assert low[1] == -np.inf and high[1] == np.inf


def test_joint_limit_rejection_preserves_source():
    cfg = NumericSanityConfig(joint_limits={"state.arm": [-1, 1]}, on_limit_violation="drop_episode")
    w = work([0, 5])
    run_episode_stages(w, numeric_checks(cfg, dataset()), numeric_transforms(cfg, dataset()))
    assert w.dropped
    assert w.keep_indices == [0, 1]
    assert np.stack(w.df["observation.state"])[1, 0] == 5


def test_interpolation_and_inf_clipping_are_transforms_only():
    cfg = NumericSanityConfig(on_nan="interpolate", on_inf="clip", outlier_mode="off")
    w = work([0, np.nan, 2, np.inf])
    NumericSanityRule(cfg, dataset()).apply(w)
    np.testing.assert_equal(np.stack(w.df["observation.state"])[:, 0], [0, 1, 2, 0])
    assert w.keep_indices == [0, 1, 2, 3]


def test_interpolation_only_changes_bad_components():
    cfg = NumericSanityConfig(on_nan="interpolate", on_inf="interpolate", outlier_mode="off")
    w = work([[0, 10], [np.nan, 99], [2, np.inf], [3, 40]])
    rule = NumericSanityRule(cfg, dataset())
    rule.apply(w)
    np.testing.assert_equal(np.stack(w.df["observation.state"]), [[0, 10], [1, 99], [2, 69.5], [3, 40]])
    assert w.keep_indices == [0, 1, 2, 3]
