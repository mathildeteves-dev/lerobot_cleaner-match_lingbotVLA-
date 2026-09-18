"""Conservative, row-preserving LeRobot v3.0 audit and numeric cleaning.

No episode/frame removal, resampling or video transformation is performed.
The legacy memory engine loads all rows with a guard; engine="streaming"
dispatches to the bounded-batch implementation used by the DROID config.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Alias(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class V3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nonfinite: Literal["error", "interpolate"] = "error"
    # Bounds apply only when explicitly requested. No inferred joint limits.
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    aliases: list[Alias] = Field(default_factory=list)
    verify_videos: bool = False
    max_frames: int = Field(default=1_000_000, gt=0)
    engine: Literal["memory", "streaming"] = "memory"
    data_file_policy: Literal["strict", "metadata_referenced"] = "strict"
    batch_rows: int = Field(default=8192, ge=1, le=65536)
    metadata_batch_rows: int = Field(default=64, ge=1, le=1024)
    max_episode_frames: int = Field(default=100_000, ge=1)
    quantile_samples: int = Field(default=32768, ge=1, le=1_000_000)
    progress: bool = False
    disk_reserve_gb: float = Field(default=5.0, ge=0)

    @model_validator(mode="after")
    def validate_file_policy(self):
        if self.data_file_policy != "strict" and self.engine != "streaming":
            raise ValueError("metadata_referenced requires engine=streaming")
        return self

    @classmethod
    def from_yaml(cls, path: Path | None) -> V3Config:
        return (
            cls()
            if path is None
            else cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
        )


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


def load_v3(root: Path, max_frames: int = 1_000_000):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("This command requires LeRobot v3.0; use run for GR00T v2.1")
    if not 0 < info["total_frames"] <= max_frames:
        raise ValueError(f"Empty dataset or total_frames exceeds max_frames={max_frames}")
    if not np.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError("fps must be positive and finite")
    ep_paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not ep_paths:
        raise ValueError("Missing meta/episodes parquet files")
    ep_tables = [(p, pd.read_parquet(p)) for p in ep_paths]
    episodes = pd.concat([t for _, t in ep_tables], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    data_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    parts = [(p, pd.read_parquet(p)) for p in data_paths]
    if not parts:
        raise ValueError("Missing data parquet files")
    frame_count = sum(len(t) for _, t in parts)
    if frame_count != info["total_frames"] or frame_count > max_frames:
        raise ValueError("Actual row count differs from info.total_frames or exceeds limit")
    data = pd.concat([t for _, t in parts], ignore_index=True)
    required = set(info["features"]) - {
        k for k, v in info["features"].items() if v["dtype"] == "video"
    }
    if required - set(data):
        raise ValueError(f"Missing feature columns: {sorted(required - set(data))}")
    if len(episodes) != info["total_episodes"] or not np.array_equal(
        episodes.episode_index, np.arange(len(episodes))
    ):
        raise ValueError("Episode count/indices are inconsistent")
    if not np.array_equal(data["index"], np.arange(len(data))):
        raise ValueError("Global data index must be contiguous in parquet file order")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    if len(tasks) != info["total_tasks"] or not np.array_equal(
        tasks.task_index, np.arange(len(tasks))
    ):
        raise ValueError("tasks.parquet indices/count are inconsistent")
    if not set(data.task_index).issubset(set(tasks.task_index)):
        raise ValueError("Data references undefined task indices")
    for key in numeric_keys(info):
        if matrix(data[key]).shape[1] != int(np.prod(info["features"][key]["shape"])):
            raise ValueError(f"Feature dimension mismatch: {key}")
    videos = {}
    cursor = 0
    file_ranges = {}
    for path, table in parts:
        file_ranges[path.resolve()] = (cursor, cursor + len(table))
        cursor += len(table)
    cursor = 0
    for _, ep in episodes.iterrows():
        lo, hi = int(ep.dataset_from_index), int(ep.dataset_to_index)
        if lo != cursor or hi <= lo or hi - lo != int(ep.length) or hi > len(data):
            raise ValueError(f"Invalid episode interval: {ep.episode_index}")
        cursor = hi
        group = data.iloc[lo:hi]
        if not (group.episode_index == ep.episode_index).all():
            raise ValueError("Episode metadata does not match data rows")
        if not np.array_equal(group.frame_index, np.arange(len(group))):
            raise ValueError("Non-contiguous frame_index")
        if not np.allclose(group.timestamp, np.arange(len(group)) / info["fps"], atol=1e-4, rtol=0):
            raise ValueError("Non-uniform timestamps: refusing to relabel time without resampling")
        path = safe_path(
            root,
            info["data_path"].format(
                chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"])
            ),
        )
        if path not in file_ranges or not file_ranges[path][0] <= lo < file_ranges[path][1]:
            raise ValueError("Episode data file reference does not contain its first row")
        for key, feature in info["features"].items():
            if feature["dtype"] != "video":
                continue
            prefix = f"videos/{key}/"
            path = safe_path(
                root,
                info["video_path"].format(
                    video_key=key,
                    chunk_index=int(ep[prefix + "chunk_index"]),
                    file_index=int(ep[prefix + "file_index"]),
                ),
            )
            start, end = float(ep[prefix + "from_timestamp"]), float(ep[prefix + "to_timestamp"])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing/empty referenced video: {path}")
            if (
                not np.isfinite([start, end]).all()
                or start < 0
                or not np.isclose(end - start, len(group) / info["fps"], atol=1e-3)
            ):
                raise ValueError(f"Invalid episode video interval: {key}, {ep.episode_index}")
            item = videos.setdefault(
                str(path.relative_to(root)),
                {
                    "expected_end": 0.0,
                    "intervals": [],
                    "key": key,
                },
            )
            if item["intervals"] and start < item["expected_end"] - 1e-3:
                raise ValueError(f"Overlapping video intervals: {key}")
            item["expected_end"] = end
            item["intervals"].append([start, end])
    if cursor != len(data):
        raise ValueError("Unreferenced trailing data rows")
    return root, info, data, episodes, parts, ep_tables, videos


def verify_video_files(root: Path, info: dict, videos: dict) -> None:
    try:
        import av
    except ImportError as e:
        raise ImportError("Video verification requires pip install -e '.[v3-video]'") from e
    for relative, item in videos.items():
        times = []
        with av.open(str(root / relative)) as container:
            stream = container.streams.video[0]
            shape = info["features"][item["key"]]["shape"]
            for frame in container.decode(stream):
                if [frame.height, frame.width, 3] != shape or frame.pts is None:
                    raise ValueError(f"Video dimensions/PTS invalid: {relative}")
                times.append(float(frame.pts * stream.time_base))
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


def audit_v3(dataset: Path, config: V3Config | None = None) -> dict:
    config = config or V3Config()
    if config.engine == "streaming":
        from lerobot_cleaner.v30.v3_streaming import audit_streaming

        return audit_streaming(dataset, config)
    root, info, data, episodes, _, _, videos = load_v3(dataset, config.max_frames)
    if config.verify_videos:
        verify_video_files(root, info, videos)
    return {
        "dataset": str(root),
        "version": "v3.0",
        "robot_type": info.get("robot_type"),
        "episodes": len(episodes),
        "frames": len(data),
        "fps": info["fps"],
        "tasks": info["total_tasks"],
        "numeric": describe(data, info),
        "videos": videos,
        "video_verification": "full_decode" if config.verify_videos else "metadata_only",
        "unsuccessful_episodes": int(
            data.groupby("episode_index")["is_episode_successful"].first().eq(False).sum()
        )
        if "is_episode_successful" in data
        else None,
        "warning": "No rows are removed. Numeric checks do not prove task success or control semantics.",
    }


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


def clean_v3(
    dataset: Path, output: Path, config: V3Config | None = None, *, resume: bool = False
) -> dict:
    config = config or V3Config()
    if config.engine == "streaming":
        from lerobot_cleaner.v30.v3_streaming import clean_streaming

        return clean_streaming(dataset, output, config, resume=resume)
    if resume:
        raise ValueError("resume is supported only by the streaming engine")
    root, info, data, episodes, parts, ep_tables, videos = load_v3(dataset, config.max_frames)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be a new directory outside the input dataset")
    if config.verify_videos:
        verify_video_files(root, info, videos)
    before = describe(data, info)
    changes = {}
    keys = numeric_keys(info)
    alias_targets = {a.target for a in config.aliases}
    if len(alias_targets) != len(config.aliases):
        raise ValueError("Duplicate alias targets")
    if alias_targets & set(config.bounds):
        raise ValueError("Set bounds on canonical source, not on an alias target")
    for alias in config.aliases:
        if alias.source not in keys or alias.target not in keys or alias.source in alias_targets:
            raise ValueError("Alias must reference numeric columns without chained/cyclic aliases")
        source, target = matrix(data[alias.source]), matrix(data[alias.target])
        if (
            not 0 <= alias.start < alias.end <= source.shape[1]
            or alias.end - alias.start != target.shape[1]
        ):
            raise ValueError("Invalid alias slice")
        if not np.allclose(source[:, alias.start : alias.end], target, equal_nan=True):
            raise ValueError(f"Alias mismatch in source dataset: {alias.source} -> {alias.target}")
    for key, (low, high) in config.bounds.items():
        if key not in keys or not info["features"][key]["dtype"].startswith("float"):
            raise ValueError(f"Bounds require a floating-point feature: {key}")
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"Invalid bounds: {key}")
    for key in keys:
        if key in alias_targets:
            continue
        arr = matrix(data[key]).astype(np.float64)
        original = arr.copy()
        if not np.isfinite(arr).all():
            if config.nonfinite == "error" or key in {
                "timestamp",
                "index",
                "frame_index",
                "episode_index",
                "task_index",
            }:
                raise ValueError(f"Non-finite values in {key}; no output was written")
            for _, ep in episodes.iterrows():
                lo, hi = int(ep.dataset_from_index), int(ep.dataset_to_index)
                for dim in range(arr.shape[1]):
                    values = arr[lo:hi, dim]
                    valid = np.isfinite(values)
                    if not valid.any():
                        raise ValueError(
                            f"Cannot interpolate entirely invalid {key} in episode {ep.episode_index}"
                        )
                    x = np.arange(len(values))
                    values[~valid] = np.interp(x[~valid], x[valid], values[valid])
        if key in config.bounds:
            arr = np.clip(arr, *config.bounds[key])
        count = int((~np.isclose(arr, original, rtol=0, atol=0, equal_nan=True)).sum())
        if count:
            put_matrix(data, key, arr, info["features"][key]["dtype"])
            changes[key] = count
    for alias in config.aliases:
        values = matrix(data[alias.source])[:, alias.start : alias.end]
        count = int(
            (~np.isclose(values, matrix(data[alias.target]), rtol=0, atol=0, equal_nan=True)).sum()
        )
        if count:
            put_matrix(data, alias.target, values, info["features"][alias.target]["dtype"])
            changes[alias.target] = count
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    ep_stats = {}
    for key in keys:
        arr = matrix(data[key])
        stats[key] = feature_stats(arr)
        for _, ep in episodes.iterrows():
            values = feature_stats(arr[int(ep.dataset_from_index) : int(ep.dataset_to_index)])
            ep_stats.setdefault(int(ep.episode_index), {}).update(
                {f"stats/{key}/{name}": value for name, value in values.items()}
            )
    report = {
        "input": str(root),
        "output": str(output),
        "version": "v3.0",
        "episodes": len(episodes),
        "frames": len(data),
        "changed_values": changes,
        "rows_removed": 0,
        "video_verification": "full_decode" if config.verify_videos else "metadata_only",
        "numeric_before": before,
        "numeric_after": describe(data, info),
        "policy": "Preserve every row, task, timestamp, video offset and extra feature; no success filtering.",
        "video_sha256": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clean-v3-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        # Copy standard components only; source images/ placeholders and local junk are not data.
        shutil.copytree(root / "meta", stage / "meta")
        cursor = 0
        for path, table in parts:
            dest = stage / path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            data.iloc[cursor : cursor + len(table)].to_parquet(dest, index=False)
            cursor += len(table)
        for path, table in ep_tables:
            updated = pd.DataFrame(
                [ep_stats[int(i)] for i in table.episode_index], index=table.index
            )
            table = pd.concat(
                [table.drop(columns=updated.columns, errors="ignore"), updated], axis=1
            )
            table.to_parquet(stage / path.relative_to(root), index=False)
        (stage / "meta/stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        for relative in videos:
            dest = stage / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, dest)

            def digest(path):
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                        h.update(block)
                return h.hexdigest()

            source_hash = digest(root / relative)
            if digest(dest) != source_hash:
                raise RuntimeError("Copied video checksum mismatch")
            report["video_sha256"][relative] = source_hash
        load_v3(stage, config.max_frames)
        report_dir = stage / "cleaning_report"
        report_dir.mkdir()
        (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (report_dir / "cleaning_config.used.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        (report_dir / "report.md").write_text(
            f"# LeRobot v3 cleaning report\n\nEpisodes: {len(episodes)}; frames: {len(data)}.\n\n"
            f"Changed values: {changes}. No frames removed.\n\n"
            f"Video verification: {report['video_verification']}. Videos copied byte-for-byte (SHA256 checked).\n\n"
            "LingBot normalization must still be computed with its own compute_norm.py.\n",
            encoding="utf-8",
        )
        stage.rename(output)
    return report
