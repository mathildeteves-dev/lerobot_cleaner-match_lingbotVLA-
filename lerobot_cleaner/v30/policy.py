"""Checked policy configuration; findings and mutations remain separate."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RulePolicy(StrictModel):
    on_fail: Literal["report", "warn", "reject_episode", "abort"] = "warn"
    on_unevaluated: Literal["report", "warn", "reject_episode", "abort"] = "warn"


RULE_NAMES = {"finite", "timestamp", "episode_structure", "episode_length", "video_integrity",
              "velocity", "acceleration", "jerk", "velocity_zscore", "acceleration_zscore",
              "static_ratio", "joint_static_ratio", "static_edges", "joint_limits", "zscore",
              "percentile_outlier", "gripper", "blur", "roi", "aliases"}


class QualityPolicy(StrictModel):
    default: RulePolicy = Field(default_factory=RulePolicy)
    rules: dict[str, RulePolicy] = Field(default_factory=dict)
    # Input decisions drive mutation; output decisions gate publication, never trigger a loop.
    output_on_fail: Literal["report", "abort"] = "report"

    @model_validator(mode="after")
    def valid_rules(self):
        for key in self.rules:
            if key.rsplit("/", 1)[-1] not in RULE_NAMES:
                raise ValueError(f"Unknown quality policy rule: {key}")
        return self


class StaticTrimPolicy(StrictModel):
    enabled: bool = False
    source: Literal["state", "action"] = "action"
    columns: list[int] | None = None
    epsilon: float = Field(.002, ge=0)
    min_kept_frames: int = Field(2, ge=1)
    mode: Literal["trim_edges", "drop_static_frames"] = "trim_edges"


class FrameFilterPolicy(StrictModel):
    nonfinite: bool = False
    # Original episode-relative indices, never updated indices.
    drop_frames: dict[int, list[int]] = Field(default_factory=dict)


class NumericTransform(StrictModel):
    target: str
    operation: Literal["interpolate", "clip", "percentile_clip", "gripper"]
    low: float | list[float] | None = None
    high: float | list[float] | None = None
    quantiles: tuple[float, float] = (.01, .99)
    threshold: float = .5
    mode: Literal["binary_01", "binary_neg1_pos1"] = "binary_01"

    @model_validator(mode="after")
    def required_bounds(self):
        if self.operation == "clip" and (self.low is None or self.high is None):
            raise ValueError("clip requires explicit reference low/high bounds")
        if self.operation == "percentile_clip":
            if (self.low is None) != (self.high is None):
                raise ValueError("Provide both percentile bounds, or neither for source calibration")
            if not 0 <= self.quantiles[0] < self.quantiles[1] <= 1:
                raise ValueError("Invalid percentile quantiles")
        return self


class CropPolicy(StrictModel):
    purpose: Literal["dataset", "preprocessing"] = "dataset"
    # Pixel coordinates [left, top, right, bottom], relative to the original frame.
    box: tuple[int, int, int, int]
    resize: tuple[int, int] | None = None

    @model_validator(mode="after")
    def valid_box(self):
        left, top, right, bottom = self.box
        if min(left, top) < 0 or right <= left or bottom <= top:
            raise ValueError("ROI must be a nonempty pixel box")
        if self.resize is not None and min(self.resize) < 1:
            raise ValueError("resize must be positive [width, height]")
        return self


class MutationPolicy(StrictModel):
    static_trim: StaticTrimPolicy = Field(default_factory=StaticTrimPolicy)
    frame_filter: FrameFilterPolicy = Field(default_factory=FrameFilterPolicy)
    reject_episodes: list[int] = Field(default_factory=list)
    numeric: list[NumericTransform] = Field(default_factory=list)
    crop: dict[str, CropPolicy] = Field(default_factory=dict)
    retime: bool = False
    min_kept_frames: int = Field(1, ge=1)
    on_too_short: Literal["reject_episode", "abort"] = "reject_episode"

    @model_validator(mode="after")
    def valid_indices(self):
        if any(i < 0 for i in self.reject_episodes):
            raise ValueError("episode IDs must be nonnegative")
        for eid, positions in self.frame_filter.drop_frames.items():
            if eid < 0 or any(i < 0 for i in positions) or len(positions) != len(set(positions)):
                raise ValueError("drop_frames requires unique nonnegative source positions")
        cols = self.static_trim.columns
        if cols is not None and (not cols or min(cols) < 0 or len(cols) != len(set(cols))):
            raise ValueError("Static columns must be nonempty, unique and nonnegative")
        return self
