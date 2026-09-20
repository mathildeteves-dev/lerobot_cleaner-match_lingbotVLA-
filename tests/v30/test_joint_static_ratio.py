"""Joint static uses the intersection of same-transition masks, not two ratios."""
import numpy as np
import pandas as pd
import pytest
from test_audit_quality import fingerprints

from lerobot_cleaner.core import TrajectoryView
from lerobot_cleaner.core.quality import check_joint_static_ratio
from lerobot_cleaner.v30.quality import TrajectoryQualityConfig, audit_trajectory
from lerobot_cleaner.v30.v3 import V3Config, audit_v3


def trajectory(state, action):
    s = np.asarray(state, dtype=float).reshape(len(state), -1)
    a = np.asarray(action, dtype=float).reshape(len(action), -1)
    return TrajectoryView(s, a, np.arange(len(s), dtype=float), 1)


def test_disjoint_static_masks_do_not_count_as_joint_static():
    view = trajectory([0, 0, 1, 1, 2], [0, 1, 1, 2, 2])
    result = check_joint_static_ratio(view, .1, .1)
    assert result.passed
    assert result.metrics["state_static_transitions"] == 2
    assert result.metrics["action_static_transitions"] == 2
    assert result.metrics["static_ratio"] == 0


@pytest.mark.parametrize("state,action", [([0, 0, 0], [0, 1, 2]), ([0, 1, 2], [0, 0, 0])])
def test_either_side_moving_prevents_static(state, action):
    result = check_joint_static_ratio(trajectory(state, action), .1, .1)
    assert result.metrics["static_ratio"] == 0


def test_fully_static_episode_exceeds_default_threshold():
    result = check_joint_static_ratio(trajectory([0, 0, 0], [0, 0, 0]), .1, .2)
    assert not result.passed and result.severity == "warning"
    assert result.metrics["static_ratio"] == 1


def test_strict_epsilon_and_ratio_threshold_boundaries():
    result = check_joint_static_ratio(trajectory([0, 1, 1], [0, 0, 0]), 1, 1, threshold=.5)
    assert result.metrics["static_transitions"] == 1  # delta == epsilon is not static
    assert result.passed  # ratio == threshold is allowed
    zero = check_joint_static_ratio(trajectory([0, 0], [0, 0]), 0, 0)
    assert zero.metrics["static_ratio"] == 0


def test_different_widths_and_independent_epsilons():
    view = trajectory([[0, 0], [0, .2], [0, .4]], [0, .05, .1])
    assert check_joint_static_ratio(view, .3, .1).metrics["static_ratio"] == 1
    assert check_joint_static_ratio(view, .1, .3).metrics["static_ratio"] == 0


@pytest.mark.parametrize("view", [trajectory([0], [0]), trajectory([0, np.nan], [0, 0])])
def test_unevaluable_inputs_are_not_passes(view):
    result = check_joint_static_ratio(view, .1, .1)
    assert not result.passed and result.metrics["evaluated"] is False


@pytest.mark.parametrize("kwargs", [{"state_epsilon": -1}, {"action_epsilon": np.inf}, {"threshold": 1.1}])
def test_invalid_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        check_joint_static_ratio(trajectory([0, 1], [0, 1]), **{"state_epsilon": .1, "action_epsilon": .1, **kwargs})


def test_audit_adds_one_joint_result_alongside_groups():
    data = pd.DataFrame({"episode_index": [0]*5, "timestamp": np.arange(5),
        "observation.state": list(np.array([0, 0, 1, 1, 2])[:, None]),
        "action": list(np.array([0, 1, 1, 2, 2])[:, None])})
    config = TrajectoryQualityConfig(groups=[{"name": "arm", "source": "state", "columns": [0]}],
                                    joint_static_ratio={"state_epsilon": .1, "action_epsilon": .1})
    rec = audit_trajectory(data, 1, config)
    assert rec["checks"]["joint_static_ratio"]["metrics"]["static_ratio"] == 0
    assert "joint_static_ratio" not in rec["groups"]["arm"]["checks"]


def test_joint_check_is_readonly_and_matches_both_audit_engines(v3_data):
    before = fingerprints(v3_data)
    cfg = V3Config(quality={"joint_static_ratio": {"state_epsilon": .01, "action_epsilon": .02}})
    memory = audit_v3(v3_data, cfg)["trajectory_quality"]
    streaming = audit_v3(v3_data, cfg.model_copy(update={"engine": "streaming"}))["trajectory_quality"]
    assert memory == streaming
    assert fingerprints(v3_data) == before
