"""Read-only audit options and per-episode dispatch to the shared core."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lerobot_cleaner.core.quality import (
    check_acceleration,
    check_acceleration_zscore,
    check_finite,
    check_jerk,
    check_static_ratio,
    check_velocity,
    check_velocity_zscore,
)

from .adapter import V30Adapter


class TrajectoryQualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    state_column: str = "observation.state"
    action_column: str = "action"
    source: Literal["state", "action"] = "state"
    columns: list[int] | None = None
    velocity: float | None = Field(None, gt=0)
    acceleration: float | None = Field(None, gt=0)
    jerk: float | None = Field(None, gt=0)
    velocity_zscore: float = Field(3.0, gt=0)
    acceleration_zscore: float = Field(3.0, gt=0)
    static_epsilon: float = Field(1e-4, ge=0)
    static_ratio: float | None = Field(None, ge=0, le=1)


def audit_trajectory(frame, fps, config):
    view = V30Adapter.to_trajectory(frame, fps, config.state_column, config.action_column)
    columns = config.columns
    width = getattr(view, config.source).shape[1]
    if columns is not None and (not columns or len(set(columns)) != len(columns) or any(i < 0 or i >= width for i in columns)):
        raise ValueError("quality.columns must contain unique in-range component indices")
    kwargs = {"source": config.source, "columns": columns}
    results = [check_finite(view),
        check_velocity(view, config.velocity, **kwargs),
        check_acceleration(view, config.acceleration, **kwargs),
        check_jerk(view, config.jerk, **kwargs),
        check_velocity_zscore(view, config.velocity_zscore, **kwargs),
        check_acceleration_zscore(view, config.acceleration_zscore, **kwargs),
        check_static_ratio(view, config.static_epsilon, config.static_ratio, **kwargs)]
    return {"episode_index": int(frame.episode_index.iloc[0]),
            "checks": {value.rule: value.to_dict() for value in results}}
