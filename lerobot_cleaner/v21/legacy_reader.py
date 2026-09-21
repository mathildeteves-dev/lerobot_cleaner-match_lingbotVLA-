"""Read a GR00T-format LeRobot dataset: meta files, modality map, episodes.

This module never mutates anything on disk. It exposes:
  - LeRobotDataset: handles to meta + per-episode parquet/video paths
  - ModalityResolver: turns ``state.left_gripper`` into concrete column slices
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from lerobot_cleaner.adapters.episode import EpisodeRef

from lerobot_cleaner.adapters.base import DatasetAdapter
from lerobot_cleaner.adapters.schema import FeatureSchema, FeatureSlice

from lerobot_cleaner.adapters.modality import (
    META_DIR, INFO_FILE, MODALITY_FILE, EPISODES_FILE, TASKS_FILE, STATS_FILE,
    RELATIVE_STATS_FILE, STATE_COL, ACTION_COL, ModalityResolver, ModalitySlice,
)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LegacyGrootAdapter(DatasetAdapter):
    """Read-only handle to a GR00T-format LeRobot dataset directory."""

    def __init__(self, root: str | Path, *, writer=None):
        self._writer = writer
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.meta_dir = self.root / META_DIR
        self._load_meta()
        self._episode_lookup = {ref.episode_index: ref for ref in self.episodes()}

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

    def _features(self, modality):
        column = STATE_COL if modality == "state" else ACTION_COL
        return tuple(FeatureSchema(f"{modality}.{key}", modality,
                             (FeatureSlice(column, int(block["start"]), int(block["end"])),))
                     for key, block in self.modality.get(modality, {}).items())

    def get_state_features(self):
        return self._features("state")

    def get_action_features(self):
        return self._features("action")

    def get_camera_features(self):
        return tuple(FeatureSchema(key, "video", camera_column=self.resolver.video_original_key(key))
                     for key in self.resolver.video_keys())

    def read_episode(self, episode_id):
        ref = self._episode_lookup.get(episode_id)
        if ref is None:
            raise KeyError(f"Unknown episode: {episode_id}")
        return self.from_frame(ref.load_parquet(), ref=ref)

    # Preserve the historical raw-vector layout in the canonical schema.
    raw_vectors = True


def load_legacy_parquet(path):
    """Explicit legacy-only physical access; not used by the official pipeline."""
    import pandas as pd
    return pd.read_parquet(path)
