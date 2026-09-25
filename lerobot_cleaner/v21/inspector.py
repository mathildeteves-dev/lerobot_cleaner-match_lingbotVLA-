"""Scan a dataset and produce a human/machine summary (used by wizard + report)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from lerobot_cleaner.v21.reader import LeRobotDataset


@dataclass
class DatasetSummary:
    root: str
    codebase_version: str
    robot_type: str
    fps: float
    total_episodes: int
    total_frames: int
    total_tasks: int
    state_dim: int
    action_dim: int
    state_keys: dict[str, list[int]]  # key -> [start, end]
    action_keys: dict[str, list[int]]
    video_keys: list[str]
    video_shapes: dict[str, list[int]]  # key -> [h, w, c]
    video_codecs: dict[str, str]
    episode_length_min: int
    episode_length_max: int
    episode_length_mean: float
    tasks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_dataset(ds: LeRobotDataset) -> DatasetSummary:
    lengths = [int(e.get("length", 0)) for e in ds.episodes_meta] or [0]

    state_keys = {k: [v["start"], v["end"]] for k, v in ds.modality.get("state", {}).items()}
    action_keys = {k: [v["start"], v["end"]] for k, v in ds.modality.get("action", {}).items()}

    video_shapes: dict[str, list[int]] = {}
    video_codecs: dict[str, str] = {}
    for k in ds.video_keys:
        feat = ds.features[k]
        video_shapes[k] = list(feat.get("shape", []))
        video_codecs[k] = feat.get("info", {}).get("video.codec", "unknown")

    return DatasetSummary(
        root=str(ds.root),
        codebase_version=ds.info.get("codebase_version", "unknown"),
        robot_type=ds.info.get("robot_type", "unknown"),
        fps=ds.fps,
        total_episodes=int(ds.info.get("total_episodes", len(ds.episodes_meta))),
        total_frames=int(ds.info.get("total_frames", sum(lengths))),
        total_tasks=int(ds.info.get("total_tasks", len(ds.tasks_meta))),
        state_dim=ds.resolver.state_dim(),
        action_dim=ds.resolver.action_dim(),
        state_keys=state_keys,
        action_keys=action_keys,
        video_keys=ds.resolver.video_keys(),
        video_shapes=video_shapes,
        video_codecs=video_codecs,
        episode_length_min=min(lengths),
        episode_length_max=max(lengths),
        episode_length_mean=sum(lengths) / len(lengths),
        tasks=[t.get("task", "") for t in ds.tasks_meta],
    )
