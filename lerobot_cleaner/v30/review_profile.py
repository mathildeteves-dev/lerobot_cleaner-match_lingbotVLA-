"""Dataset contracts and explicitly heuristic review thresholds."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lerobot_cleaner.v30.v3 import Alias


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VisualOptions(StrictModel):
    enabled: bool = True
    sample_seconds: float = Field(default=1, gt=0)
    black_level: float = Field(default=3, ge=0, le=255)
    low_detail_variance: float = Field(default=4, ge=0)
    unchanged_pixel_tolerance: float = Field(default=0.05, ge=0)
    unchanged_seconds: float = Field(default=3, gt=0)
    preview_limit: int = Field(default=20, ge=0, le=1000)


class QualityOptions(StrictModel):
    motion_action_dims: list[int] | None = None
    static_epsilon: float = Field(default=0.0001, ge=0)
    static_seconds: float = Field(default=3, gt=0)
    action_jump: float = Field(default=0.5, gt=0)
    jitter_step: float = Field(default=0.1, ge=0)
    jitter_fraction: float = Field(default=0.25, ge=0, le=1)
    min_seconds: float = Field(default=2, ge=0)
    max_seconds: float = Field(default=120, gt=0)
    visual: VisualOptions = Field(default_factory=VisualOptions)

    @model_validator(mode="after")
    def check_lengths(self):
        if self.min_seconds >= self.max_seconds:
            raise ValueError("min_seconds must be less than max_seconds")
        return self


class Semantics(StrictModel):
    verified: bool = False
    evidence: str = ""
    action_mode: str = "unknown"
    rotation_representation: str = "unknown"
    units: str = "unknown"
    gripper_convention: str = "unknown"

    @model_validator(mode="after")
    def require_evidence(self):
        if self.verified and any(
            not value.strip() or value.lower() == "unknown"
            for key, value in self.model_dump().items()
            if key != "verified"
        ):
            raise ValueError("Verified semantics require evidence and every convention")
        return self


class ReviewProfile(StrictModel):
    name: str
    features: dict[str, int]
    cameras: list[str]
    state_feature: str
    action_feature: str
    aliases: list[Alias] = Field(default_factory=list)
    alias_atol: float = Field(default=1e-6, ge=0)
    alias_rtol: float = Field(default=1e-6, ge=0)
    quality: QualityOptions = Field(default_factory=QualityOptions)
    semantics: Semantics = Field(default_factory=Semantics)

    @model_validator(mode="after")
    def check_contract(self):
        if not self.features or any(v < 1 for v in self.features.values()):
            raise ValueError("Feature widths must be positive")
        if self.state_feature not in self.features or self.action_feature not in self.features:
            raise ValueError("State/action feature must appear in profile.features")
        dims = self.quality.motion_action_dims
        if dims is not None and (
            not dims
            or len(set(dims)) != len(dims)
            or any(d < 0 or d >= self.features[self.action_feature] for d in dims)
        ):
            raise ValueError("Invalid motion_action_dims")
        if not self.cameras or len(self.cameras) != len(set(self.cameras)):
            raise ValueError("Provide unique cameras")
        for alias in self.aliases:
            if (
                alias.source not in self.features
                or alias.target not in self.features
                or not 0 <= alias.start < alias.end <= self.features[alias.source]
                or alias.end - alias.start != self.features[alias.target]
            ):
                raise ValueError("Invalid profile alias slice")
        return self


DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "configs/profiles/libero_fastwam.yaml"


def load_profile(path=None):
    return ReviewProfile.model_validate(
        yaml.safe_load(Path(path or DEFAULT_PROFILE).read_text(encoding="utf-8"))
    )
