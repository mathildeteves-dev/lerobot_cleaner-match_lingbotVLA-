"""Per-component-group quality with independent source, thresholds, and results."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from test_audit_quality import fingerprints

from lerobot_cleaner.v30.quality import TrajectoryQualityConfig, audit_trajectory
from lerobot_cleaner.v30.review_profile import load_profile
from lerobot_cleaner.v30.v3 import V3Config, audit_v3


def groups():
    return [
        {"name": "state_arm", "source": "state", "columns": [0], "velocity": 1},
        {"name": "action_arm", "source": "action", "columns": [0], "velocity": 1},
        {"name": "state_gripper", "source": "state", "columns": [1], "velocity": 100},
        {"name": "action_gripper", "source": "action", "columns": [1], "velocity": 100},
    ]


def frame():
    state = np.zeros((30, 2))
    action = np.zeros((30, 2))
    action[15:, 0] = 5
    state[15:, 1] = 1
    return pd.DataFrame({"episode_index": np.zeros(30, dtype=int), "timestamp": np.arange(30)/10,
                         "observation.state": list(state), "action": list(action)})


def test_action_spike_is_checked_without_contaminating_other_groups():
    data = frame()
    before = data.copy(deep=True)
    configured = groups()
    configured[1].update(velocity_zscore=2, acceleration_zscore=2, static_epsilon=.2)
    configured[0].update(velocity_zscore=6, acceleration_zscore=6, static_epsilon=.001)
    result = audit_trajectory(data, 10, TrajectoryQualityConfig(groups=configured))
    assert set(result["groups"]) == {item["name"] for item in configured}
    state = result["groups"]["state_arm"]["checks"]
    action = result["groups"]["action_arm"]["checks"]
    assert state["velocity"]["passed"]
    assert not action["velocity"]["passed"]
    assert not action["velocity_zscore"]["passed"]
    assert not action["acceleration_zscore"]["passed"]
    assert result["groups"]["state_gripper"]["checks"]["velocity"]["passed"]
    assert state["velocity_zscore"]["metrics"]["threshold"] == 6
    assert action["velocity_zscore"]["metrics"]["threshold"] == 2
    assert action["static_ratio"]["metrics"]["epsilon"] == .2
    pd.testing.assert_frame_equal(data, before)


def test_nonfinite_gripper_does_not_fail_arm_group():
    data = frame()
    data.at[0, "action"] = np.array([0, np.nan])
    result = audit_trajectory(data, 10, TrajectoryQualityConfig(groups=groups()))
    assert not result["checks"]["finite"]["passed"]
    assert result["groups"]["action_arm"]["checks"]["finite"]["passed"]
    assert not result["groups"]["action_gripper"]["checks"]["finite"]["passed"]


@pytest.mark.parametrize("invalid", [
    [], [{"name": "x", "source": "state", "columns": []}],
    [{"name": "x", "source": "state", "columns": [0, 0]}],
    [{"name": "x", "source": "state", "columns": [-1]}],
    [{"name": "x", "source": "state", "columns": [True]}],
    [{"name": "x", "source": "bad", "columns": [0]}],
    [{"name": "x", "source": "state", "columns": [0]}]*2,
])
def test_invalid_group_config_fails_early(invalid):
    with pytest.raises(ValidationError):
        TrajectoryQualityConfig(groups=invalid)


def test_mixed_settings_and_out_of_range_columns_are_not_ignored():
    with pytest.raises(ValidationError, match="inside each group"):
        TrajectoryQualityConfig(groups=groups(), velocity=2)
    config = TrajectoryQualityConfig(groups=[{"name": "bad", "source": "action", "columns": [2]}])
    with pytest.raises(ValueError, match="bad.*width"):
        audit_trajectory(frame(), 10, config)


def test_group_config_roundtrip_and_legacy_single_source():
    config = TrajectoryQualityConfig(groups=groups())
    assert TrajectoryQualityConfig.model_validate_json(config.model_dump_json()) == config
    old = audit_trajectory(frame(), 10, TrajectoryQualityConfig(source="action", columns=[0], velocity=1))
    assert old["mode"] == "legacy_single_source"
    assert not old["checks"]["velocity"]["passed"]


def test_group_audit_matches_across_engines_and_preserves_files(v3_data):
    before = fingerprints(v3_data)
    config = V3Config(quality={"groups": groups()})
    memory = audit_v3(v3_data, config)["trajectory_quality"]
    streaming = audit_v3(v3_data, config.model_copy(update={"engine": "streaming"}))["trajectory_quality"]
    assert memory == streaming
    assert fingerprints(v3_data) == before


def test_shipped_configs_have_robot_specific_groups():
    root = Path(__file__).resolve().parents[2]
    droid = V3Config.from_yaml(root / "configs/cleaning/droid_v3.yaml")
    assert droid.quality.groups[3].columns == [7]
    libero = V3Config.from_yaml(root / "configs/cleaning/libero_v3.yaml")
    assert libero.quality.groups[2].columns == [6, 7]
    assert libero.quality.groups[3].columns == [6]
    profile = load_profile(root / "configs/profiles/libero_fastwam.yaml")
    assert profile.quality.groups == libero.quality.groups
