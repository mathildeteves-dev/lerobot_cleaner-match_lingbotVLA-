"""Version-3 quality settings; no mutation settings belong here."""
from typing import Literal
from pydantic import Field, model_validator
from .policy import StrictModel


class LanguageCheck(StrictModel):
    enabled: bool = True
    allow_instruction_changes: bool = False
    min_length: int | None = Field(2, ge=0)
    max_length: int | None = Field(4096, ge=1)
    check_control_characters: bool = True
    check_numeric: bool = True
    placeholders: list[str] = Field(default_factory=lambda: ["n/a", "unknown", "task"])

    @model_validator(mode="after")
    def ordered(self):
        if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
            raise ValueError("Invalid task text length range")
        return self


class BoundsCheck(StrictModel):
    low: float | list[float]
    high: float | list[float]


class TimestampCheck(StrictModel):
    enabled: bool = True
    tolerance_ratio: float = Field(.2, ge=0)
    require_uniform: bool = True
    require_zero_start: bool = True


class LengthCheck(StrictModel):
    enabled: bool = True
    min_frames: int = Field(1, ge=0)
    max_frames: int | None = Field(None, ge=1)
    min_seconds: float | None = Field(None, ge=0)
    max_seconds: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.max_frames is not None and self.max_frames < self.min_frames:
            raise ValueError("Invalid frame length range")
        if self.min_seconds is not None and self.max_seconds is not None and self.min_seconds > self.max_seconds:
            raise ValueError("Invalid duration range")
        return self


class GripperCheck(StrictModel):
    source: Literal["state", "action"]
    columns: list[int] = Field(min_length=1)
    mode: Literal["continuous", "binary_01", "binary_neg1_pos1"] = "continuous"
    low: float = 0.
    high: float = 1.
    tolerance: float = Field(1e-6, ge=0)

    @model_validator(mode="after")
    def valid(self):
        if self.low > self.high or min(self.columns) < 0 or len(self.columns) != len(set(self.columns)):
            raise ValueError("Invalid gripper limits/columns")
        return self


class VisualCheck(StrictModel):
    enabled: bool = True  # Metadata references are checked without decoding by default.
    decode: bool = False
    sample_stride: int = Field(30, ge=1)
    batch_frames: int = Field(32, ge=1, le=256)
    blur_variance: float | None = Field(None, ge=0)
    max_blur_ratio: float = Field(.5, ge=0, le=1)
    black_level: float | None = Field(None, ge=0, le=255)
    max_black_ratio: float = Field(.5, ge=0, le=1)
    min_brightness: float | None = Field(None, ge=0, le=255)
    max_brightness: float | None = Field(None, ge=0, le=255)

    @model_validator(mode="after")
    def needs_pixels(self):
        if any(v is not None for v in (self.blur_variance, self.black_level, self.min_brightness, self.max_brightness)) and not self.decode:
            raise ValueError("Visual pixel checking requires quality.visual.decode=true")
        if self.min_brightness is not None and self.max_brightness is not None and self.min_brightness > self.max_brightness:
            raise ValueError("Invalid brightness range")
        return self


VideoCheck = VisualCheck  # Legacy import alias.
