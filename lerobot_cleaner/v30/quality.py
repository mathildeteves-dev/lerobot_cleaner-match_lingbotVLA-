"""Explicit component groups for read-only trajectory quality audits."""
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lerobot_cleaner.core import TrajectoryView
from lerobot_cleaner.core.quality import (
    check_acceleration,
    check_acceleration_zscore,
    check_finite,
    check_jerk,
    check_joint_static_ratio,
    check_static_ratio,
    check_velocity,
    check_velocity_zscore,
)

from .adapter import V30Adapter

ColumnIndex = Annotated[int, Field(strict=True, ge=0)]


class QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    velocity: float | None = Field(None, gt=0)
    acceleration: float | None = Field(None, gt=0)
    jerk: float | None = Field(None, gt=0)
    velocity_zscore: float = Field(3.0, gt=0)
    acceleration_zscore: float = Field(3.0, gt=0)
    static_epsilon: float = Field(1e-4, ge=0)
    static_ratio: float | None = Field(None, ge=0, le=1)


class QualityGroup(QualityThresholds):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    source: Literal["state", "action"]
    columns: list[ColumnIndex] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_columns(self):
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("quality group columns must be unique")
        return self


class JointStaticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    state_epsilon: float = Field(ge=0)
    action_epsilon: float = Field(ge=0)
    threshold: float = Field(0.95, ge=0, le=1)


class TrajectoryQualityConfig(QualityThresholds):
    joint_static_ratio: JointStaticConfig | None = None
    enabled: bool = True
    state_column: str = "observation.state"
    action_column: str = "action"
    groups: list[QualityGroup] | None = Field(None, min_length=1)
    # Legacy single-source configuration, used only when groups is absent.
    source: Literal["state", "action"] = "state"
    columns: list[ColumnIndex] | None = None

    @model_validator(mode="after")
    def validate_groups(self):
        if self.groups is not None:
            names = [group.name for group in self.groups]
            if len(set(names)) != len(names):
                raise ValueError("quality group names must be unique")
            legacy_fields = set(QualityThresholds.model_fields) | {"source", "columns"}
            # Serialized default fields remain compatible with round-trips. Explicit
            # non-default legacy settings are rejected instead of silently ignored.
            defaults = QualityThresholds()
            conflicts = [key for key in legacy_fields if key in self.model_fields_set
                         and getattr(self, key) != (getattr(defaults, key) if key in QualityThresholds.model_fields else {"source": "state", "columns": None}[key])]
            if conflicts:
                raise ValueError(f"With quality.groups, place per-group settings inside each group: {conflicts}")
        return self


def _checks(view, settings, source, columns):
    kwargs = {"source": source, "columns": columns}
    results = [check_velocity(view, settings.velocity, **kwargs),
        check_acceleration(view, settings.acceleration, **kwargs),
        check_jerk(view, settings.jerk, **kwargs),
        check_velocity_zscore(view, settings.velocity_zscore, **kwargs),
        check_acceleration_zscore(view, settings.acceleration_zscore, **kwargs),
        check_static_ratio(view, settings.static_epsilon, settings.static_ratio, **kwargs)]
    return {value.rule: value.to_dict() for value in results}


def audit_trajectory(frame, fps, config):
    view = V30Adapter.to_trajectory(frame, fps, config.state_column, config.action_column)
    record = {"episode_index": int(frame.episode_index.iloc[0]),
              "checks": {"finite": check_finite(view).to_dict()}}
    if config.joint_static_ratio is not None:
        joint = check_joint_static_ratio(view, **config.joint_static_ratio.model_dump())
        record["checks"][joint.rule] = joint.to_dict()
    if config.groups is None:
        columns = config.columns
        width = getattr(view, config.source).shape[1]
        if columns is not None and (not columns or len(set(columns)) != len(columns) or any(i >= width for i in columns)):
            raise ValueError("quality.columns must contain unique in-range component indices")
        record["checks"].update(_checks(view, config, config.source, columns))
        record["mode"] = "legacy_single_source"
        return record
    # Validate every group before evaluation, so a typo cannot silently omit action.
    for group in config.groups:
        width = getattr(view, group.source).shape[1]
        if any(index >= width for index in group.columns):
            raise ValueError(f"quality.groups[{group.name}] columns {group.columns} exceed {group.source} width {width}")
    record["mode"] = "groups"
    record["groups"] = {}
    for group in config.groups:
        selected = getattr(view, group.source)[:, group.columns]
        empty = np.empty((len(selected), 0))
        group_view = TrajectoryView(selected if group.source == "state" else empty,
                                    selected if group.source == "action" else empty,
                                    view.timestamps, view.fps)
        checks = {"finite": check_finite(group_view).to_dict()}
        checks.update(_checks(group_view, group, group.source, None))
        record["groups"][group.name] = {
            "source": group.source, "columns": group.columns,
            "thresholds": group.model_dump(exclude={"name", "source", "columns"}),
            "checks": checks,
        }
    return record
