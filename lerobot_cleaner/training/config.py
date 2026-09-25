"""Opt-in training diagnostics, independent of quality and mutation policies."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class TokenizationCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # Directory or cached model ID. No network download is performed.
    tokenizer_path: str | None = None


class TrainingCheckConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["lingbot"] = "lingbot"
    tokenization: TokenizationCheckConfig = Field(default_factory=TokenizationCheckConfig)
    robot_config: Path
    train_config: Path
    entrypoint: Literal["official_train", "explicit_dataset"] = "official_train"
    padding_warning_ratio: float | None = Field(None, ge=0, le=1)
    include_padding_by_timestep: bool = False
    decode_cameras: bool = False
    camera_sample_stride: int = Field(30, ge=1)
