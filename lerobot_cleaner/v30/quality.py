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
from .quality_groups import QualityGroupResolver
from lerobot_cleaner.adapters.resolver import FeatureResolver
from .quality_options import LanguageCheck, BoundsCheck, TimestampCheck, LengthCheck, GripperCheck, VisualCheck
from lerobot_cleaner.core.quality.joint_limits import check_joint_limits
from lerobot_cleaner.core.quality.zscore import check_zscore
from lerobot_cleaner.core.quality.percentile import check_percentile
from lerobot_cleaner.core.quality.integrity.timestamp import check_timestamp
from lerobot_cleaner.core.quality.integrity.episode_structure import check_episode_structure, check_episode_length
from lerobot_cleaner.core.quality.embodiment.gripper import check_gripper

ColumnIndex = Annotated[int, Field(strict=True, ge=0)]


class QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    joint_limits: BoundsCheck | None = None
    percentile_bounds: BoundsCheck | None = None
    zscore: float | None = Field(None, gt=0)
    velocity: float | None = Field(None, gt=0)
    acceleration: float | None = Field(None, gt=0)
    jerk: float | None = Field(None, gt=0)
    velocity_zscore: float = Field(3.0, gt=0)
    acceleration_zscore: float = Field(3.0, gt=0)
    static_epsilon: float = Field(1e-4, ge=0)
    static_ratio: float | None = Field(None, ge=0, le=1)


class QualityGroup(QualityThresholds):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    feature: str | None = Field(None, strict=True, min_length=1)
    features: list[Annotated[str, Field(strict=True, min_length=1)]] | None = Field(None, min_length=1)
    source: Literal["state", "action"] | None = None
    columns: list[ColumnIndex] | None = Field(None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def source_reference(cls, data):
        if isinstance(data, dict) and isinstance(data.get("source"), dict):
            nested = data["source"]
            if len(nested) != 1 or next(iter(nested)) not in {"feature", "features"}:
                raise ValueError(f"Quality group {data.get('name')!r}: source must reference feature or features")
            if data.get("feature") is not None or data.get("features") is not None:
                raise ValueError(f"Quality group {data.get('name')!r}: conflicting feature selectors")
            data = {**data, "source": None, **nested}
        if isinstance(data, dict) and data.get("features") is not None and not isinstance(data["features"], (list, tuple)):
            raise ValueError(f"Quality group {data.get('name')!r}: canonical features must be an ordered list")
        return data

    @model_validator(mode="after")
    def unique_selection(self):
        prefix = f"Quality group {self.name!r}: "
        semantic = int(self.feature is not None) + int(self.features is not None)
        legacy = self.source is not None or self.columns is not None
        if semantic > 1 or (semantic and legacy):
            raise ValueError(prefix + "choose feature, features, or legacy source + columns exclusively")
        if not semantic:
            if self.source is None or self.columns is None:
                raise ValueError(prefix + "provide feature/features or both source and columns")
            if len(set(self.columns)) != len(self.columns):
                raise ValueError(prefix + "columns must be unique")
        else:
            requested = [self.feature] if self.feature is not None else self.features
            for i, name in enumerate(requested):
                if not name.strip() or name in requested[:i]:
                    raise ValueError(prefix + f"canonical feature {name!r} is empty or repeated")
        return self


class JointStaticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    state_epsilon: float = Field(ge=0)
    action_epsilon: float = Field(ge=0)
    threshold: float = Field(0.95, ge=0, le=1)


class TrajectoryQualityConfig(QualityThresholds):
    language: LanguageCheck = Field(default_factory=LanguageCheck)
    timestamp: TimestampCheck = Field(default_factory=TimestampCheck)
    episode_structure: bool = True
    episode_length: LengthCheck = Field(default_factory=LengthCheck)
    gripper: GripperCheck | None = None
    visual: VisualCheck = Field(default_factory=VisualCheck)
    joint_static_ratio: JointStaticConfig | None = None
    enabled: bool = True
    state_column: str = "observation.state"
    action_column: str = "action"
    groups: list[QualityGroup] | None = Field(None, min_length=1)
    # Legacy single-source configuration, used only when groups is absent.
    source: Literal["state", "action"] = "state"
    columns: list[ColumnIndex] | None = None

    @model_validator(mode="before")
    @classmethod
    def visual_alias(cls, data):
        if isinstance(data, dict) and "video" in data:
            if "visual" in data:
                raise ValueError("Use quality.visual or legacy quality.video, not both")
            data = dict(data)
            data["visual"] = data.pop("video")
        return data

    @property
    def video(self):
        return self.visual

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
    width = getattr(view, source).shape[1] if columns is None else len(columns)
    for bounds, checker in [(settings.joint_limits, check_joint_limits),
                            (settings.percentile_bounds, check_percentile)]:
        if bounds is not None:
            low, high = np.asarray(bounds.low), np.asarray(bounds.high)
            if (low.ndim > 1 or high.ndim > 1 or low.size not in {1, width}
                    or high.size not in {1, width} or np.any(low > high)):
                raise ValueError("Bounds must be ordered scalars or match selected dimensions")
            results.append(checker(view, bounds.low, bounds.high, **kwargs))
    if settings.zscore is not None:
        results.append(check_zscore(view, settings.zscore, **kwargs))
    return {value.rule: value.to_dict() for value in results}


def audit_trajectory(frame, fps, config, *, adapter=None, metadata=None, schema=None):
    if adapter is not None:
        if schema is not None:
            raise ValueError("Pass adapter or schema, not both")
        episode = adapter.from_frame(frame)
        schema = episode.feature_schema
        view = episode.to_trajectory()
    elif schema is not None:
        from lerobot_cleaner.adapters.builder import EpisodeBuilder
        view = EpisodeBuilder(schema, fps).build(frame, metadata=metadata).to_trajectory()
    else:
        view = V30Adapter.to_trajectory(frame, fps, config.state_column, config.action_column)
    schema = schema or FeatureResolver.default_schema(view.state.shape[1], view.action.shape[1],
                                                      config.state_column, config.action_column)
    selections = ([QualityGroupResolver().resolve(group, schema,
                  {"state": view.state.shape[1], "action": view.action.shape[1]}) for group in config.groups]
                  if config.groups is not None else [])
    record = {"episode_index": int(frame.episode_index.iloc[0]),
              "checks": {"finite": check_finite(view).to_dict()}}
    if config.timestamp.enabled:
        record["checks"]["timestamp"] = check_timestamp(view, **config.timestamp.model_dump(exclude={"enabled"})).to_dict()
    if config.episode_structure:
        record["checks"]["episode_structure"] = check_episode_structure(
            {name: frame[name].to_numpy() for name in frame.columns if name in
             {"index", "episode_index", "frame_index", "timestamp", "task_index"}},
            len(frame), record["episode_index"], metadata).to_dict()
    if config.episode_length.enabled:
        record["checks"]["episode_length"] = check_episode_length(
            view, **config.episode_length.model_dump(exclude={"enabled"})).to_dict()
    if config.gripper is not None:
        if max(config.gripper.columns) >= getattr(view, config.gripper.source).shape[1]:
            raise ValueError("Gripper columns exceed canonical feature dimensions")
        record["checks"]["gripper"] = check_gripper(view, **config.gripper.model_dump()).to_dict()
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
    record["mode"] = "groups"
    record["groups"] = {}
    for group, selection in zip(config.groups, selections):
        resolved = selection["resolved"]
        source, columns = resolved["source"], resolved["columns"]
        selected = getattr(view, source)[:, columns]
        empty = np.empty((len(selected), 0))
        group_view = TrajectoryView(selected if source == "state" else empty,
                                    selected if source == "action" else empty,
                                    view.timestamps, view.fps,
                                    tuple(view.state_semantics[i] for i in columns) if source == "state" else (),
                                    tuple(view.action_semantics[i] for i in columns) if source == "action" else ())
        checks = {"finite": check_finite(group_view).to_dict()}
        try:
            checks.update(_checks(group_view, group, source, None))
        except ValueError as exc:
            raise ValueError(f"Quality group {group.name!r}, canonical features {selection['requested_features']!r}: {exc}") from exc
        record["groups"][group.name] = {
            **selection, "source": source, "columns": columns,
            "thresholds": group.model_dump(exclude={"name", "source", "columns", "feature", "features"}),
            "checks": checks,
        }
    return record
