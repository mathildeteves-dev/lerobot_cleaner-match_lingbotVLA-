"""V3 configuration and public entry points for the quality/plan/transform pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lerobot_cleaner.adapters.factory import v3_adapter
from lerobot_cleaner.v30.quality import TrajectoryQualityConfig, audit_trajectory
from lerobot_cleaner.v30.policy import QualityPolicy, MutationPolicy
from lerobot_cleaner.training.config import TrainingCheckConfig


class Alias(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class V3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reader_backend: Literal["lerobot"] = "lerobot"
    semantic_adapter: Literal["auto", "generic", "lingbot", "lerobot", "groot"] = "auto"
    converted_root: Path | None = None
    modality_config: Path | None = None
    robot_config: Path | None = None
    mapping_config: Path | None = None
    training_check: TrainingCheckConfig | None = None
    quality: TrajectoryQualityConfig = Field(default_factory=TrajectoryQualityConfig)
    policy: QualityPolicy = Field(default_factory=QualityPolicy)
    transforms: MutationPolicy = Field(default_factory=MutationPolicy)
    nonfinite: Literal["error", "interpolate"] = "error"
    # Bounds apply only when explicitly requested. No inferred joint limits.
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    aliases: list[Alias] = Field(default_factory=list)
    verify_videos: bool = False
    max_frames: int = Field(default=1_000_000, gt=0)
    engine: Literal["memory", "streaming"] = "memory"
    data_file_policy: Literal["strict"] = "strict"
    batch_rows: int = Field(default=8192, ge=1, le=65536)
    metadata_batch_rows: int = Field(default=64, ge=1, le=1024)
    max_episode_frames: int = Field(default=100_000, ge=1)
    quantile_samples: int = Field(default=32768, ge=1, le=1_000_000)
    progress: bool = False
    disk_reserve_gb: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def validate_file_policy(self):
        for name in ("robot_config", "modality_config", "mapping_config", "converted_root"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.resolve())
        if self.semantic_adapter == "lingbot" and self.robot_config is None:
            raise ValueError("LingBot semantics require robot_config")
        if self.semantic_adapter == "generic" and self.mapping_config is None:
            raise ValueError("Generic semantics require mapping_config")
        if self.semantic_adapter == "auto" and sum(value is not None for value in
                (self.robot_config, self.modality_config, self.mapping_config)) > 1:
            raise ValueError("Select semantic_adapter explicitly when multiple mappings are configured")
        if self.mapping_config is not None and self.semantic_adapter not in {"auto", "generic"}:
            raise ValueError("mapping_config requires semantic_adapter=generic or auto")
        if not self.quality.enabled and self.policy.rules:
            raise ValueError("Per-rule policies require quality.enabled=true")
        targets = [entry.target for entry in self.aliases]
        if len(targets) != len(set(targets)):
            raise ValueError("Duplicate alias targets")
        if any(item.target in targets for item in self.transforms.numeric):
            raise ValueError("Transform canonical source features; alias targets synchronize afterward")
        return self

    @classmethod
    def from_yaml(cls, path: Path | None) -> V3Config:
        if path is None:
            return cls()
        path = Path(path).resolve()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name in ("robot_config", "modality_config", "mapping_config", "converted_root"):
            if data.get(name) is not None:
                value = Path(data[name])
                data[name] = str((path.parent / value).resolve() if not value.is_absolute() else value)
        if data.get("training_check"):
            for name in ("robot_config", "train_config"):
                if data["training_check"].get(name):
                    value = Path(data["training_check"][name])
                    data["training_check"][name] = str((path.parent / value).resolve() if not value.is_absolute() else value)
        return cls.model_validate(data)



def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


def matrix(series: pd.Series) -> np.ndarray:
    return np.asarray(series.tolist()).reshape(len(series), -1)


def numeric_keys(info: dict) -> list[str]:
    return [
        k
        for k, v in info["features"].items()
        if v["dtype"] == "bool" or v["dtype"].startswith(("float", "int", "uint"))
    ]


def load_v3(root: Path, max_frames: int = 1_000_000, *, reader_backend="lerobot", config=None):
    """Memory view supplied exclusively by the official storage boundary."""
    from lerobot_cleaner.storage import OfficialStorage
    from lerobot_cleaner.v30.v3_streaming import validate_episode
    config = config or V3Config(max_frames=max_frames, reader_backend=reader_backend)
    with OfficialStorage(root, config) as storage:
        snapshot = storage.memory_snapshot()
        _, info, data, episodes, _, _, _ = snapshot
        offset = 0
        for row in episodes.to_dict("records"):
            start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
            if not 0 < stop - start <= config.max_episode_frames:
                raise ValueError("Episode exceeds configured frame limit")
            frame = data.iloc[start:stop]
            storage.validate_episode(row, offset)
            validate_episode(info, row, int(row["episode_index"]), offset, frame)
            offset += len(frame)
        if offset != info["total_frames"]:
            raise ValueError("Unreferenced data rows")
        return snapshot


def episode_frames(data, episodes):
    """Use only the offsets supplied by the official metadata snapshot."""
    for row in episodes.itertuples(index=False):
        yield data.iloc[int(row.dataset_from_index):int(row.dataset_to_index)]


def verify_video_files(root: Path, info: dict, videos: dict) -> None:
    from lerobot_cleaner.storage.video import decoded_video
    for relative, item in videos.items():
        times = []
        with decoded_video(root / relative) as decoded_frames:
            shape = info["features"][item["key"]]["shape"]
            for frame in decoded_frames:
                if [frame.height, frame.width, 3] != shape or frame.pts is None:
                    raise ValueError(f"Video dimensions/PTS invalid: {relative}")
                times.append(float(frame.pts * frame.time_base))
        ts = np.asarray(times)
        if len(ts) == 0 or not np.allclose(np.diff(ts), 1 / info["fps"], atol=1e-3):
            raise ValueError(f"Video decode/frame spacing failed: {relative}")
        for start, end in item["intervals"]:
            count = int(((ts >= start - 1e-4) & (ts < end - 1e-4)).sum())
            if count != round((end - start) * info["fps"]):
                raise ValueError(f"Video interval frame count mismatch: {relative}, {start}")
        item["decoded_frames"] = len(ts)


def describe(data: pd.DataFrame, info: dict) -> dict:
    result = {}
    for key in numeric_keys(info):
        arr = matrix(data[key]).astype(float)
        finite = np.isfinite(arr)
        result[key] = {"nonfinite": int((~finite).sum()), "dimensions": arr.shape[1]}
        if finite.all():
            result[key].update(min=arr.min(axis=0).tolist(), max=arr.max(axis=0).tolist())
    return result


def audit_v3(dataset: Path, config: V3Config | None = None, *, output_check=False) -> dict:
    """Read-only checks and transform plans; no transforms or dataset writes."""
    from .pipeline import audit_pipeline
    return audit_pipeline(dataset, config or V3Config(), output_check=output_check)


def feature_stats(arr: np.ndarray) -> dict:
    arr = arr.astype(np.float64)
    result = {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [len(arr)],
    }
    quantiles = [1, 10, 50, 90, 99]
    for q, values in zip(quantiles, np.quantile(arr, np.asarray(quantiles) / 100, axis=0)):
        result[f"q{q:02d}"] = values.tolist()
    return result


def put_matrix(data: pd.DataFrame, key: str, arr: np.ndarray, dtype: str) -> None:
    arr = arr.astype(dtype)
    scalar = np.isscalar(data[key].iloc[0])
    data[key] = arr[:, 0] if scalar else list(arr)


def clean_v3(dataset: Path, output: Path, config: V3Config | None = None, *, resume=False) -> dict:
    """Execute explicit plans, write official v3 data, finalize and recheck output."""
    from .pipeline import clean_pipeline
    return clean_pipeline(dataset, output, config or V3Config(), resume=resume)
