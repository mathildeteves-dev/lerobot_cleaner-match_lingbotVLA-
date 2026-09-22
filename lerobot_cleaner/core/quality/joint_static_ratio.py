"""Joint state/action stillness on the same consecutive transitions."""
import numpy as np
from ..physical import PhysicalDeltaResolver

from ._common import result


def check_joint_static_ratio(trajectory, state_epsilon, action_epsilon, threshold=0.95):
    """Flag excessive joint stillness; epsilon comparisons are strictly less-than.

    All state components AND all action components must be stationary on the
    same transition. A ratio greater than threshold is a quality warning.
    State/action widths may differ, but transition counts must match.
    """
    for name, epsilon in (("state_epsilon", state_epsilon), ("action_epsilon", action_epsilon)):
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("joint static threshold must be within [0, 1]")
    metrics = {"evaluated": False, "state_epsilon": state_epsilon,
               "action_epsilon": action_epsilon, "threshold": threshold}
    state, action = trajectory.state, trajectory.action
    if len(state) < 2 or not state.size or not action.size:
        return result("joint_static_ratio", False, metrics, "insufficient state/action trajectory")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        return result("joint_static_ratio", False, metrics, "non-finite state/action trajectory")
    try:
        state_steps = PhysicalDeltaResolver.trajectory_difference(trajectory, "state")
        action_steps = PhysicalDeltaResolver.trajectory_difference(trajectory, "action")
    except ValueError as exc:
        return result("joint_static_ratio", False, metrics, str(exc))
    with np.errstate(over="ignore", invalid="ignore"):
        state_delta = np.max(np.abs(state_steps), axis=1)
        action_delta = np.max(np.abs(action_steps), axis=1)
    if not np.isfinite(state_delta).all() or not np.isfinite(action_delta).all():
        return result("joint_static_ratio", False, metrics, "non-finite trajectory differences")
    state_static = state_delta < state_epsilon
    action_static = action_delta < action_epsilon
    static = state_static & action_static
    ratio = float(static.mean())
    metrics.update(evaluated=True, static_ratio=ratio, static_transitions=int(static.sum()),
                   state_static_transitions=int(state_static.sum()),
                   action_static_transitions=int(action_static.sum()), transitions=len(static))
    return result("joint_static_ratio", ratio <= threshold, metrics,
                  "joint static ratio exceeds threshold" if ratio > threshold else None)
