"""Cross-version parity and dependency boundary of trajectory quality."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lerobot_cleaner.core import TrajectoryView
from lerobot_cleaner.core.quality import (
    check_acceleration,
    check_finite,
    check_jerk,
    check_joint_limits,
    check_velocity,
    check_zscore,
)
from lerobot_cleaner.v21.adapter import V21Adapter
from lerobot_cleaner.v30.adapter import V30Adapter


def adapters():
    t = np.arange(8) * .2
    frame = pd.DataFrame({"observation.state": list((t**3)[:, None]),
                          "action": list((2*t)[:, None]), "timestamp": t,
                          "episode_index": [7]*len(t)})
    work = SimpleNamespace(df=frame, keep_indices=list(range(len(t))))
    return V21Adapter.to_trajectory(work, 5), V30Adapter.to_trajectory(frame, 5)


@pytest.mark.parametrize("check,kwargs", [
    (check_finite, {}), (check_velocity, {"threshold": 1}),
    (check_acceleration, {"threshold": 1}), (check_jerk, {"threshold": 5}),
    (check_joint_limits, {"low": -1, "high": 1}), (check_zscore, {"threshold": 1}),
])
def test_identical_trajectory_has_identical_quality_across_versions(check, kwargs):
    v21, v30 = adapters()
    assert check(v21, **kwargs).to_dict() == check(v30, **kwargs).to_dict()


def test_jerk_is_cubic_derivative_in_seconds():
    view, _ = adapters()
    result = check_jerk(view, threshold=5)
    assert not result.passed
    assert result.metrics["max_jerk"] == pytest.approx(6)


def test_adapters_preserve_frame_gaps_when_timestamps_absent():
    frame = pd.DataFrame({"observation.state": [np.array([0]), np.array([2])],
                          "action": [np.array([0]), np.array([0])], "frame_index": [4, 6]})
    a = V21Adapter.to_trajectory(SimpleNamespace(df=frame, keep_indices=[4, 6]), 10)
    b = V30Adapter.to_trajectory(frame, 10)
    np.testing.assert_allclose(a.timestamps, [.4, .6])
    assert check_velocity(a).to_dict() == check_velocity(b).to_dict()


def test_view_owns_readonly_arrays():
    state = np.ones((4, 1))
    view = TrajectoryView(state, state, np.arange(4), 1)
    state[0, 0] = 99
    assert view.state[0, 0] == 1
    with pytest.raises(ValueError):
        view.state[0, 0] = 3
    assert check_zscore(view).metrics["max_zscore"] == 0


def test_invalid_shapes_and_mixed_episodes_rejected():
    with pytest.raises(ValueError, match="equal frame"):
        TrajectoryView(np.zeros((2, 1)), np.zeros((3, 1)), np.arange(2), 10)
    with pytest.raises(ValueError, match="single episode"):
        V30Adapter.to_trajectory(pd.DataFrame({"episode_index": [0, 1]}), 10)


def test_nonfinite_and_invalid_timestamps_are_not_passes():
    view = TrajectoryView(np.array([[0], [np.nan], [1], [2]]), np.zeros((4, 1)), np.array([0, 0, 1, 2]), 1)
    assert not check_finite(view).passed
    assert not check_jerk(view).passed
    assert not check_zscore(view).passed


def test_core_has_no_version_or_storage_imports():
    root = Path(__file__).resolve().parents[1] / "lerobot_cleaner/core"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            assert not any(name.startswith(("lerobot_cleaner.v21", "lerobot_cleaner.v30", "pandas", "pyarrow")) for name in names)
