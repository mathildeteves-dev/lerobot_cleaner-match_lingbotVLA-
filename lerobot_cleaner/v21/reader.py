"""Read a GR00T-format LeRobot dataset: meta files, modality map, episodes.

This module never mutates anything on disk. It exposes:
  - LeRobotDataset: handles to meta + per-episode parquet/video paths
  - ModalityResolver: turns ``state.left_gripper`` into concrete column slices
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

META_DIR = "meta"
INFO_FILE = "info.json"
MODALITY_FILE = "modality.json"
EPISODES_FILE = "episodes.jsonl"
TASKS_FILE = "tasks.jsonl"
STATS_FILE = "stats.json"
RELATIVE_STATS_FILE = "relative_stats.json"

# Standard parquet columns that carry the state/action vectors.
STATE_COL = "observation.state"
ACTION_COL = "action"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class EpisodeRef:
    """Locator + metadata for a single episode in the source dataset."""

    episode_index: int
    length: int
    tasks: list[str]
    parquet_path: Path
    # video_key (original_key) -> path
    video_paths: dict[str, Path] = field(default_factory=dict)

    def load_parquet(self) -> pd.DataFrame:
        return pd.read_parquet(self.parquet_path)


@dataclass
class ModalitySlice:
    """A named contiguous slice within the state/action vector."""

    modality: str  # "state" | "action"
    key: str  # e.g. "left_gripper"
    start: int
    end: int

    @property
    def column(self) -> str:
        return STATE_COL if self.modality == "state" else ACTION_COL


class ModalityResolver:
    """Resolve dotted modality keys (``state.left_gripper``) to vector slices."""

    def __init__(self, modality_meta: dict[str, Any]):
        self.meta = modality_meta

    def resolve(self, dotted_key: str) -> ModalitySlice:
        if "." not in dotted_key:
            raise ValueError(
                f"Modality target '{dotted_key}' must be of the form "
                f"'<state|action>.<key>', e.g. 'state.left_gripper'."
            )
        modality, key = dotted_key.split(".", 1)
        if modality not in ("state", "action"):
            raise ValueError(f"Unknown modality '{modality}' in '{dotted_key}'.")
        if modality not in self.meta or key not in self.meta[modality]:
            available = list(self.meta.get(modality, {}).keys())
            raise KeyError(
                f"Key '{key}' not in modality.json[{modality}]. Available: {available}"
            )
        block = self.meta[modality][key]
        return ModalitySlice(modality, key, int(block["start"]), int(block["end"]))

    def video_keys(self) -> list[str]:
        return list(self.meta.get("video", {}).keys())

    def video_original_key(self, key: str) -> str:
        return self.meta["video"][key].get("original_key", key)

    def state_dim(self) -> int:
        return max((b["end"] for b in self.meta.get("state", {}).values()), default=0)

    def action_dim(self) -> int:
        return max((b["end"] for b in self.meta.get("action", {}).values()), default=0)

    def arm_keys(self, modality: str = "state") -> list[str]:
        """Keys whose name contains 'arm' — used for relative_stats."""
        return [k for k in self.meta.get(modality, {}) if "arm" in k.lower()]


class LeRobotDataset:
    """Read-only handle to a GR00T-format LeRobot dataset directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.meta_dir = self.root / META_DIR
        self._load_meta()

    # --- meta loading -------------------------------------------------------
    def _load_meta(self) -> None:
        self.info = self._load_json(INFO_FILE, required=True)
        self.modality = self._load_json(MODALITY_FILE, required=True)
        self.resolver = ModalityResolver(self.modality)
        self.stats = self._load_json(STATS_FILE, required=False) or {}
        self.relative_stats = self._load_json(RELATIVE_STATS_FILE, required=False) or {}
        self.episodes_meta = _read_jsonl(self.meta_dir / EPISODES_FILE)
        tasks_path = self.meta_dir / TASKS_FILE
        self.tasks_meta = _read_jsonl(tasks_path) if tasks_path.exists() else []

        self.fps: float = float(self.info.get("fps", 30))
        self.chunk_size: int = int(self.info.get("chunks_size", 1000))
        self.data_path_pattern: str = self.info["data_path"]
        self.video_path_pattern: Optional[str] = self.info.get("video_path")
        self.features: dict[str, Any] = self.info.get("features", {})

    def _load_json(self, name: str, required: bool) -> Optional[dict]:
        path = self.meta_dir / name
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required meta file: {path}")
            return None
        with open(path, "r") as f:
            return json.load(f)

    # --- episode access -----------------------------------------------------
    def episodes(self) -> list[EpisodeRef]:
        refs = []
        for ep in self.episodes_meta:
            idx = int(ep["episode_index"])
            chunk = idx // self.chunk_size
            parquet_path = self.root / self.data_path_pattern.format(
                episode_chunk=chunk, episode_index=idx
            )
            video_paths = {}
            if self.video_path_pattern:
                for vkey in self.resolver.video_keys():
                    original = self.resolver.video_original_key(vkey)
                    video_paths[original] = self.root / self.video_path_pattern.format(
                        episode_chunk=chunk, video_key=original, episode_index=idx
                    )
            refs.append(
                EpisodeRef(
                    episode_index=idx,
                    length=int(ep.get("length", 0)),
                    tasks=list(ep.get("tasks", [])),
                    parquet_path=parquet_path,
                    video_paths=video_paths,
                )
            )
        return refs

    @property
    def video_keys(self) -> list[str]:
        """Original video keys present in features (dtype == 'video')."""
        return [k for k, v in self.features.items() if v.get("dtype") == "video"]
