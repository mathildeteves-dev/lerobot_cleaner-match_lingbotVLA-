"""Quality audit is read-only and equivalent in both v3 engines."""
import hashlib

import numpy as np
import pytest

from lerobot_cleaner.core import TrajectoryView
from lerobot_cleaner.core.quality import (
    check_acceleration_zscore,
    check_static_ratio,
    check_velocity_zscore,
)
from lerobot_cleaner.v30.v3 import V3Config, audit_v3


def fingerprints(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_audit_engines_share_quality_without_mutating_input(v3_data):
    before = fingerprints(v3_data)
    memory = audit_v3(v3_data, V3Config())
    streaming = audit_v3(v3_data, V3Config(engine="streaming"))
    assert memory["trajectory_quality"] == streaming["trajectory_quality"]
    assert set(memory["trajectory_quality"][0]["checks"]) == {
        "finite", "velocity", "acceleration", "jerk", "static_ratio", "velocity_zscore", "acceleration_zscore"}
    assert fingerprints(v3_data) == before


def test_derivative_zscore_detects_relative_spike_and_constant_motion():
    t = np.arange(30, dtype=float)
    linear = TrajectoryView(t[:, None], t[:, None], t, 1)
    assert check_velocity_zscore(linear).metrics["max_zscore"] == 0
    assert check_acceleration_zscore(linear).metrics["max_zscore"] == 0
    state = t.copy()
    state[15:] += 20
    spike = TrajectoryView(state[:, None], state[:, None], t, 1)
    assert not check_velocity_zscore(spike).passed
    assert not check_acceleration_zscore(spike).passed
    assert check_static_ratio(linear).metrics["static_ratio"] == 0
    static = TrajectoryView(np.zeros((30, 1)), np.zeros((30, 1)), t, 1)
    assert check_static_ratio(static).metrics["static_ratio"] == 1


def test_short_and_nonfinite_derivatives_are_not_false_passes():
    view = TrajectoryView(np.array([[np.nan], [0.]]), np.zeros((2, 1)), np.arange(2), 1)
    assert not check_velocity_zscore(view).passed
    assert not check_acceleration_zscore(view).metrics["evaluated"]
    with pytest.raises(ValueError):
        check_static_ratio(view, epsilon=-1)
