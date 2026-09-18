"""Configuration schema for lerobot-cleaner.

The schema is intentionally permissive about *which* rules exist (each rule owns
its own config block) but strict about the shared run-level options. Every rule
config inherits from :class:`RuleConfig` and is validated by pydantic, so a typo
in a yaml key fails loudly instead of being silently ignored.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuleConfig(BaseModel):
    """Base for every rule's config block."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


# --- R1: timestamp alignment ------------------------------------------------
class OnMismatch(str, Enum):
    warn_only = "warn_only"
    strict_drop = "strict_drop"


class TimestampAlignmentConfig(RuleConfig):
    expected_fps_tolerance_ratio: float = Field(
        0.2, ge=0, description="Allowed deviation of mean dt from 1/fps, as a ratio."
    )
    video_frame_tolerance: int = Field(
        2, ge=0, description="Allowed |video_frame_count - parquet_rows|."
    )
    require_uniform_dt: bool = Field(
        True,
        description="Flag episodes whose inter-frame dt is non-uniform. GR00T's "
        "action chunking assumes uniform dt between consecutive rows.",
    )
    on_mismatch: OnMismatch = OnMismatch.warn_only


# --- R2: static frame trim --------------------------------------------------
class StaticTrimMode(str, Enum):
    trim_edges = "trim_edges"
    drop_static_frames = "drop_static_frames"


class StaticSource(str, Enum):
    action = "action"
    state = "state"


class StaticFrameTrimConfig(RuleConfig):
    mode: StaticTrimMode = StaticTrimMode.trim_edges
    source: StaticSource = StaticSource.action
    pos_threshold: float = Field(
        2.0e-3, gt=0, description="Per-step L-inf motion below this counts as static."
    )
    rot_threshold_deg: Optional[float] = Field(
        2.0, description="Rotation threshold in degrees; null to disable rotation check."
    )
    # Which modality.json sub-keys hold rotation (quaternion/euler) dims, if any.
    rotation_keys: list[str] = Field(default_factory=list)
    min_kept_frames: int = Field(
        2, ge=1, description="Never trim an episode below this many frames."
    )


# --- R3: gripper binarize ---------------------------------------------------
class GripperMode(str, Enum):
    binary_01 = "binary_01"
    binary_neg1_pos1 = "binary_neg1_pos1"


class GripperBinarizeConfig(RuleConfig):
    targets: list[str] = Field(
        default_factory=list,
        description="modality.json keys to binarize, e.g. state.left_gripper.",
    )
    threshold: float = 0.5
    # Optional per-target override of the global threshold.
    per_target_threshold: dict[str, float] = Field(default_factory=dict)
    mode: GripperMode = GripperMode.binary_01
    # Guard: refuse to run if the source data already looks binarized.
    skip_if_already_binary: bool = True


# --- R4: video ROI crop -----------------------------------------------------
class ROI(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_ratio: float = Field(..., ge=0, le=1)
    y_ratio: float = Field(..., ge=0, le=1)
    w_ratio: float = Field(..., gt=0, le=1)
    h_ratio: float = Field(..., gt=0, le=1)


class VideoRoiCropConfig(RuleConfig):
    rois: dict[str, ROI] = Field(default_factory=dict)
    resize_after_crop: Optional[list[int]] = Field(
        None, description="[width, height] to resize to after cropping; null to keep."
    )


# --- R5: episode length filter ----------------------------------------------
class EpisodeLengthFilterConfig(RuleConfig):
    min_frames: int = Field(30, ge=0)
    max_frames: Optional[int] = None
    min_duration_sec: Optional[float] = None
    max_duration_sec: Optional[float] = None


# --- R6: numeric sanity -----------------------------------------------------
class OnBad(str, Enum):
    drop_episode = "drop_episode"
    drop_frame = "drop_frame"
    interpolate = "interpolate"
    clip = "clip"
    warn = "warn"


class OutlierMode(str, Enum):
    off = "off"
    warn = "warn"
    clip_quantile = "clip_quantile"
    drop_frame = "drop_frame"


class DerivativeCheckConfig(RuleConfig):
    # Absolute magnitude in target units / second**order. No implicit thresholds.
    limits: dict[str, float] = Field(default_factory=dict)
    on_violation: OnMismatch = OnMismatch.warn_only

    @model_validator(mode="after")
    def _validate_limits(self):
        import math
        if self.enabled and not self.limits:
            raise ValueError("enabled derivative checks require explicit limits")
        if any(not math.isfinite(v) or v <= 0 for v in self.limits.values()):
            raise ValueError("derivative limits must be finite and positive")
        return self


class NumericSanityConfig(RuleConfig):
    velocity: DerivativeCheckConfig = Field(default_factory=DerivativeCheckConfig)
    acceleration: DerivativeCheckConfig = Field(default_factory=DerivativeCheckConfig)
    jerk: DerivativeCheckConfig = Field(default_factory=DerivativeCheckConfig)
    on_nan: OnBad = OnBad.drop_episode
    on_inf: OnBad = OnBad.drop_episode
    # joint_limits: per modality.json key -> [low, high]
    joint_limits: dict[str, list[float]] = Field(default_factory=dict)
    on_limit_violation: OnBad = OnBad.clip
    # Outlier handling (high value because GR00T default norm is min/max).
    outlier_mode: OutlierMode = OutlierMode.warn
    outlier_low_quantile: float = Field(0.01, ge=0, le=0.5)
    outlier_high_quantile: float = Field(0.99, ge=0.5, le=1.0)
    # Which modality keys to apply outlier handling to; empty = all numeric.
    outlier_targets: list[str] = Field(default_factory=list)


# --- R7: video integrity ----------------------------------------------------
class VideoIntegrityConfig(RuleConfig):
    check_decodable: bool = True
    check_frame_count_consistency: bool = True


# --- R8: reindex & restats (always on) --------------------------------------
class ReindexConfig(RuleConfig):
    enabled: bool = True  # not user-disableable; enforced in validator
    verify_alignment: bool = Field(
        True,
        description="After writing, re-open each video and assert frame_count == "
        "row_count == episodes.jsonl length.",
    )
    compute_relative_stats: bool = Field(
        True, description="Recompute meta/relative_stats.json for arm keys."
    )


# --- Top-level rules container ----------------------------------------------
class RulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_alignment: TimestampAlignmentConfig = Field(default_factory=TimestampAlignmentConfig)
    static_frame_trim: StaticFrameTrimConfig = Field(default_factory=StaticFrameTrimConfig)
    gripper_binarize: GripperBinarizeConfig = Field(default_factory=GripperBinarizeConfig)
    video_roi_crop: VideoRoiCropConfig = Field(default_factory=VideoRoiCropConfig)
    episode_length_filter: EpisodeLengthFilterConfig = Field(
        default_factory=EpisodeLengthFilterConfig
    )
    numeric_sanity: NumericSanityConfig = Field(default_factory=NumericSanityConfig)
    video_integrity: VideoIntegrityConfig = Field(default_factory=VideoIntegrityConfig)
    reindex_and_restats: ReindexConfig = Field(default_factory=ReindexConfig)


class CleaningConfig(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    input: Optional[Path] = None
    output: Optional[Path] = None
    preset: Optional[str] = None
    dry_run: bool = False
    num_workers: int = Field(8, ge=1)
    resume: bool = Field(
        False, description="Skip episodes already present in the output directory."
    )
    rules: RulesConfig = Field(default_factory=RulesConfig)

    @model_validator(mode="after")
    def _enforce_reindex(self) -> "CleaningConfig":
        # R8 must always run; it is the only rule that finalizes meta consistency.
        self.rules.reindex_and_restats.enabled = True
        return self

    # --- (de)serialization helpers -----------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "CleaningConfig":
        path = Path(path)
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls.model_validate(raw)
        if cfg.preset:
            cfg = cfg.merge_preset(load_preset(cfg.preset))
        return cfg

    def merge_preset(self, preset_rules: dict[str, Any]) -> "CleaningConfig":
        """Overlay preset recommended_rules *under* explicit user values.

        User-specified yaml always wins; the preset only fills gaps. We achieve
        this by starting from the preset and re-applying the user's own rule dict.
        """
        recommended = preset_rules.get("recommended_rules", {})
        merged_rules = recommended.copy()
        user_rules = self.rules.model_dump(exclude_defaults=True)
        for rule_name, user_block in user_rules.items():
            base = merged_rules.get(rule_name, {})
            if isinstance(base, dict) and isinstance(user_block, dict):
                base = {**base, **user_block}
            else:
                base = user_block
            merged_rules[rule_name] = base
        data = self.model_dump(exclude={"rules"})
        data["rules"] = merged_rules
        return CleaningConfig.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json", exclude_none=False)
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


# --- preset loading ---------------------------------------------------------
PRESETS_DIR = Path(__file__).resolve().parents[2] / "presets"


def load_preset(name: str) -> dict[str, Any]:
    """Load a preset yaml by name (with or without .yaml suffix)."""
    candidate = PRESETS_DIR / name
    if candidate.suffix != ".yaml":
        candidate = PRESETS_DIR / f"{name}.yaml"
    if not candidate.exists():
        available = sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Preset '{name}' not found in {PRESETS_DIR}. Available: {available}"
        )
    with open(candidate, "r") as f:
        return yaml.safe_load(f) or {}
