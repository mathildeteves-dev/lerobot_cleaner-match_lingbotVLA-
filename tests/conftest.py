"""Shared fixtures: build a tiny synthetic GR00T-format LeRobot dataset on disk."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FPS = 10
STATE_DIM = 4  # [arm0, arm1, gripper(idx2... use 3), ...] -> we'll map gripper at 3
ACTION_DIM = 4


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _write_video(path: Path, n_frames: int, w: int = 32, h: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Generate a solid-color test video of exactly n_frames using lavfi.
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s={w}x{h}:r={FPS}:d={n_frames / FPS:.4f}",
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _make_episode_df(n: int, ep_idx: int, start_global: int, inject_static_edges=False,
                     inject_nan=False) -> pd.DataFrame:
    rng = np.random.default_rng(ep_idx)
    state = rng.normal(0, 0.1, size=(n, STATE_DIM)).astype(np.float32)
    # Make it move over time so it's not all static.
    state += np.linspace(0, 1, n)[:, None].astype(np.float32)
    if inject_static_edges:
        state[:3] = state[3]   # static lead
        state[-3:] = state[-4]  # static tail
    action = state.copy()
    # gripper dim = index 3: continuous in [0, 1]
    state[:, 3] = np.clip(rng.uniform(0, 1, size=n), 0, 1)
    action[:, 3] = state[:, 3]
    if inject_nan:
        state[1, 0] = np.nan

    ts = (np.arange(n) / FPS).astype(np.float32)
    return pd.DataFrame({
        "observation.state": list(state),
        "action": list(action),
        "timestamp": ts,
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, ep_idx, dtype=np.int64),
        "index": np.arange(start_global, start_global + n, dtype=np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
    })


@pytest.fixture
def synth_dataset(tmp_path) -> Path:
    """Build a 3-episode dataset with video, return its root path."""
    if not _has_ffmpeg():
        pytest.skip("ffmpeg not available")
    root = tmp_path / "synth_ds"
    meta = root / "meta"
    meta.mkdir(parents=True)

    lengths = [40, 12, 50]  # ep1 is short (will be filtered if min_frames=30)
    video_key = "observation.images.cam"
    chunk_size = 1000

    info = {
        "codebase_version": "v2.1",
        "robot_type": "synth",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": len(lengths),
        "total_chunks": 1,
        "chunks_size": chunk_size,
        "fps": FPS,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]},
            video_key: {
                "dtype": "video", "shape": [24, 32, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.codec": "h264", "video.fps": FPS, "video.height": 24, "video.width": 32},
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))

    modality = {
        "state": {"arm": {"start": 0, "end": 3}, "gripper": {"start": 3, "end": 4}},
        "action": {"arm": {"start": 0, "end": 3}, "gripper": {"start": 3, "end": 4}},
        "video": {video_key: {"original_key": video_key}},
        "annotation": {},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=2))

    with open(meta / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "synthetic task"}) + "\n")

    g = 0
    ep_records = []
    for i, n in enumerate(lengths):
        df = _make_episode_df(
            n, i, g,
            inject_static_edges=(i == 0),
            inject_nan=(i == 2),
        )
        g += n
        chunk = i // chunk_size
        pq = root / info["data_path"].format(episode_chunk=chunk, episode_index=i)
        pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq, index=False)
        vpath = root / info["video_path"].format(
            episode_chunk=chunk, video_key=video_key, episode_index=i
        )
        _write_video(vpath, n)
        ep_records.append({"episode_index": i, "tasks": ["synthetic task"], "length": n})

    with open(meta / "episodes.jsonl", "w") as f:
        for rec in ep_records:
            f.write(json.dumps(rec) + "\n")

    # Minimal stats.json so reader doesn't complain (pipeline recomputes anyway).
    def _stat(dim):
        return {k: [0.0] * dim for k in ("mean", "std", "min", "max", "q01", "q99")}
    stats = {"observation.state": _stat(STATE_DIM), "action": _stat(ACTION_DIM),
             "timestamp": _stat(1)}
    (meta / "stats.json").write_text(json.dumps(stats))

    return root


@pytest.fixture
def v3_data(tmp_path):
    """Synthetic LeRobot v3.0 dataset with undecodable placeholder videos."""
    root = tmp_path / "source"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [8]},
        "action.joint_position": {"dtype": "float32", "shape": [7]},
        "action.gripper_position": {"dtype": "float32", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        **{
            k: {"dtype": "int64", "shape": [1]}
            for k in ["index", "frame_index", "episode_index", "task_index"]
        },
        "is_episode_successful": {"dtype": "bool", "shape": [1]},
        "language_instruction": {"dtype": "string", "shape": [1]},
    }
    cameras = ["exterior_1_left", "exterior_2_left", "wrist_left"]
    for camera in cameras:
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": [180, 320, 3]}
    info = {
        "codebase_version": "v3.0",
        "total_frames": 6,
        "total_episodes": 2,
        "total_tasks": 1,
        "fps": 15,
        "robot_type": "Franka",
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/stats.json").write_text(
        json.dumps({"observation.images.wrist_left": {"mean": [[[0.5]], [[0.5]], [[0.5]]]}}),
        encoding="utf-8",
    )
    pd.DataFrame({"task_index": [0]}, index=["pick up object"]).to_parquet(
        root / "meta/tasks.parquet"
    )
    actions = np.arange(48, dtype=np.float32).reshape(6, 8) / 100
    table = pd.DataFrame(
        {
            "action": list(actions),
            "observation.state": list(actions / 2),
            "action.joint_position": list(actions[:, :7]),
            "action.gripper_position": actions[:, 7],
            "timestamp": np.tile(np.arange(3, dtype=np.float32) / 15, 2),
            "index": np.arange(6),
            "frame_index": np.tile(np.arange(3), 2),
            "episode_index": np.repeat([0, 1], 3),
            "task_index": np.zeros(6, dtype=np.int64),
            "is_episode_successful": [True] * 3 + [False] * 3,
            "language_instruction": ["pick up object"] * 6,
            "extra_unlisted_column": ["preserve me"] * 6,
        }
    )
    table.to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    episodes = pd.DataFrame(
        {
            "episode_index": [0, 1],
            "length": [3, 3],
            "dataset_from_index": [0, 3],
            "dataset_to_index": [3, 6],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 0],
            "stats/observation.images.wrist_left/mean": [np.array([0.5]), np.array([0.6])],
        }
    )
    for camera in cameras:
        key = f"observation.images.{camera}"
        folder = root / "videos" / key / "chunk-000"
        folder.mkdir(parents=True)
        # Intentionally NOT decodable: these tests exercise metadata-only mode.
        (folder / "file-000.mp4").write_bytes(b"fake-video-bytes")
        for suffix, values in {
            "chunk_index": [0, 0],
            "file_index": [0, 0],
            "from_timestamp": [0.0, 0.2],
            "to_timestamp": [0.2, 0.4],
        }.items():
            episodes[f"videos/{key}/{suffix}"] = values
    episodes.to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    return root
