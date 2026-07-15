"""Write cleaned episodes to a new dataset dir + finalize all GR00T meta.

The writer is the only component that mutates the output tree. It guarantees the
core GR00T invariant: for every episode and every video key,

    video_frame_count == parquet_row_count == episodes.jsonl["length"]

and rewrites info.json / episodes.jsonl / tasks.jsonl / modality.json / stats.json
/ relative_stats.json so the output is a self-consistent v2.1 dataset.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lerobot_cleaner.dataset.reader import (
    ACTION_COL,
    STATE_COL,
    LeRobotDataset,
)
from lerobot_cleaner.dataset.stats import StatsCollector, compute_episode_stats
from lerobot_cleaner.types import EpisodeWork
from lerobot_cleaner.video_utils import VideoError, copy_or_reencode, count_frames


class DatasetWriter:
    def __init__(self, source: LeRobotDataset, output_root: Path, codec: str = "libx264"):
        self.source = source
        self.out = Path(output_root)
        self.codec = codec
        self.chunk_size = source.chunk_size
        self.data_pattern = source.data_path_pattern
        self.video_pattern = source.video_path_pattern
        self.meta_dir = self.out / "meta"

        state_dim = source.resolver.state_dim()
        action_dim = source.resolver.action_dim()
        rel_slices = {
            k: (source.resolver.resolve(f"state.{k}").start, source.resolver.resolve(f"state.{k}").end)
            for k in source.resolver.arm_keys("state")
        }
        self.stats = StatsCollector(state_dim, action_dim, relative_arm_slices=rel_slices)

        # Accumulated as episodes are written, in final (reindexed) order.
        self.episodes_out: list[dict] = []
        # Per-episode stats for upstream-lerobot / pi05 compatibility
        # (meta/episodes_stats.jsonl). Each entry: {episode_index, stats}.
        self.episodes_stats_out: list[dict] = []
        self.tasks_index: dict[str, int] = {}  # task text -> new task_index
        self.total_frames = 0
        self._next_global_index = 0

    # --- task table ---------------------------------------------------------
    def _task_index(self, task: str) -> int:
        if task not in self.tasks_index:
            self.tasks_index[task] = len(self.tasks_index)
        return self.tasks_index[task]

    def _record_episode(self, new_index, n, tasks, state, action, ts) -> dict:
        """Shared tail for both write paths: accumulate global + per-episode
        stats and append the episodes.jsonl / episodes_stats.jsonl records."""
        self.stats.update(state, action, ts)
        self.episodes_stats_out.append(
            {"episode_index": new_index, "stats": compute_episode_stats(state, action, ts)}
        )
        self.total_frames += n
        record = {"episode_index": new_index, "tasks": tasks, "length": n}
        self.episodes_out.append(record)
        return record

    # --- per-episode write --------------------------------------------------
    def write_episode(self, work: EpisodeWork, new_index: int) -> dict:
        """Write one cleaned episode under its new episode_index. Returns the
        episodes.jsonl record. Raises on any alignment failure."""
        df = work.df.reset_index(drop=True)
        n = len(df)
        chunk = new_index // self.chunk_size

        # --- rewrite standard index columns ---
        df = df.copy()
        df["episode_index"] = new_index
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["index"] = np.arange(self._next_global_index, self._next_global_index + n, dtype=np.int64)
        self._next_global_index += n

        tasks = work.ref.tasks or [df.get("__task__", ["unknown"])[0]]
        primary_task = tasks[0] if tasks else "unknown"
        if "task_index" in df.columns:
            df["task_index"] = self._task_index(primary_task)
        else:
            df["task_index"] = self._task_index(primary_task)

        # --- recompute timestamp from fps to keep dt uniform ---
        # Cleaning may have removed rows; rebuild a uniform clock so downstream
        # consumers that assume dt == 1/fps stay correct.
        if "timestamp" in df.columns:
            df["timestamp"] = (np.arange(n, dtype=np.float64) / self.source.fps).astype(np.float64)

        # --- write parquet ---
        parquet_path = self.out / self.data_pattern.format(
            episode_chunk=chunk, episode_index=new_index
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        # --- write videos with alignment guarantee ---
        if self.video_pattern:
            for vkey, src_path in work.ref.video_paths.items():
                transform = work.video_transforms.get(vkey)
                keep = work.keep_indices
                crop_ratio = transform.crop_ratio if transform else None
                resize_wh = transform.resize_wh if transform else None

                dst_path = self.out / self.video_pattern.format(
                    episode_chunk=chunk, video_key=vkey, episode_index=new_index
                )
                copy_or_reencode(
                    src_path,
                    dst_path,
                    keep_indices=keep,
                    crop_ratio=crop_ratio,
                    resize_wh=resize_wh,
                    codec=self.codec,
                    fps=self.source.fps,
                )

        # --- update running stats over the global concatenation ---
        state = np.stack(df[STATE_COL].to_numpy()) if STATE_COL in df.columns else np.empty((n, 0))
        action = np.stack(df[ACTION_COL].to_numpy()) if ACTION_COL in df.columns else np.empty((n, 0))
        ts = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.zeros(n)
        return self._record_episode(new_index, n, tasks, state, action, ts)

    def finalize_staged_episode(
        self,
        new_index: int,
        staged_parquet: str,
        staged_videos: dict,
        tasks: list,
    ) -> dict:
        """Finalize an episode already cleaned + re-encoded into staging.

        Rewrites index columns + uniform timestamp, copies videos into the final
        layout, and accumulates exact streaming stats. No re-encode happens here.
        """
        df = pd.read_parquet(staged_parquet).reset_index(drop=True)
        n = len(df)
        chunk = new_index // self.chunk_size

        df["episode_index"] = new_index
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["index"] = np.arange(
            self._next_global_index, self._next_global_index + n, dtype=np.int64
        )
        self._next_global_index += n
        primary_task = tasks[0] if tasks else "unknown"
        df["task_index"] = self._task_index(primary_task)
        if "timestamp" in df.columns:
            df["timestamp"] = (np.arange(n, dtype=np.float64) / self.source.fps).astype(np.float64)

        parquet_path = self.out / self.data_pattern.format(
            episode_chunk=chunk, episode_index=new_index
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        if self.video_pattern:
            for vkey, staged in staged_videos.items():
                dst_path = self.out / self.video_pattern.format(
                    episode_chunk=chunk, video_key=vkey, episode_index=new_index
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staged), str(dst_path))

        state = np.stack(df[STATE_COL].to_numpy()) if STATE_COL in df.columns else np.empty((n, 0))
        action = np.stack(df[ACTION_COL].to_numpy()) if ACTION_COL in df.columns else np.empty((n, 0))
        ts = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.zeros(n)
        return self._record_episode(new_index, n, tasks, state, action, ts)

    # --- alignment verification --------------------------------------------
    def verify_episode_alignment(self, new_index: int, expected_len: int) -> Optional[str]:
        """Re-open written videos and assert frame_count == expected_len.
        Returns an error string on mismatch, else None."""
        if not self.video_pattern:
            return None
        chunk = new_index // self.chunk_size
        for vkey in self.source.resolver.video_keys():
            original = self.source.resolver.video_original_key(vkey)
            dst_path = self.out / self.video_pattern.format(
                episode_chunk=chunk, video_key=original, episode_index=new_index
            )
            try:
                frames = count_frames(dst_path)
            except VideoError as e:
                return f"episode {new_index} video {original}: {e}"
            if frames != expected_len:
                return (
                    f"episode {new_index} video {original}: {frames} frames != "
                    f"{expected_len} parquet rows"
                )
        return None

    # --- meta finalization --------------------------------------------------
    def finalize_meta(self, updated_video_features: Optional[dict] = None) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # tasks.jsonl
        with open(self.meta_dir / "tasks.jsonl", "w") as f:
            for task, idx in sorted(self.tasks_index.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

        # episodes.jsonl
        with open(self.meta_dir / "episodes.jsonl", "w") as f:
            for rec in self.episodes_out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # episodes_stats.jsonl — per-episode stats for upstream-lerobot / pi05.
        # GR00T reads stats.json; openpi (HF lerobot, v2.1) reads this file. We
        # always emit both so the cleaned dataset trains on either framework
        # with no further processing.
        with open(self.meta_dir / "episodes_stats.jsonl", "w") as f:
            for rec in self.episodes_stats_out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # modality.json (copy verbatim; cleaning never changes dim layout)
        with open(self.meta_dir / "modality.json", "w") as f:
            json.dump(self.source.modality, f, indent=4, ensure_ascii=False)

        # stats.json / relative_stats.json
        with open(self.meta_dir / "stats.json", "w") as f:
            json.dump(self.stats.finalize_stats(), f, indent=2)
        rel = self.stats.finalize_relative_stats()
        if rel:
            with open(self.meta_dir / "relative_stats.json", "w") as f:
                json.dump(rel, f, indent=2)

        # info.json
        info = json.loads(json.dumps(self.source.info))  # deep copy
        num_chunks = (len(self.episodes_out) - 1) // self.chunk_size + 1 if self.episodes_out else 0
        info["total_episodes"] = len(self.episodes_out)
        info["total_frames"] = self.total_frames
        info["total_tasks"] = len(self.tasks_index)
        info["total_chunks"] = num_chunks
        info["splits"] = {"train": f"0:{len(self.episodes_out)}"}
        n_video_keys = len(self.source.resolver.video_keys())
        info["total_videos"] = len(self.episodes_out) * n_video_keys
        if updated_video_features:
            for key, feat_update in updated_video_features.items():
                if key in info.get("features", {}):
                    info["features"][key].update(feat_update)
        # Codec changed on re-encode.
        for key, feat in info.get("features", {}).items():
            if feat.get("dtype") == "video":
                feat.setdefault("info", {})["video.codec"] = self.codec_name()
        with open(self.meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

    def codec_name(self) -> str:
        return {"libx264": "h264", "libx265": "hevc"}.get(self.codec, self.codec)

    def existing_output_episodes(self) -> set[int]:
        """For --resume: episode indices already fully written to output."""
        done = set()
        ep_file = self.meta_dir / "episodes.jsonl"
        if not ep_file.exists():
            return done
        with open(ep_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["episode_index"])
        return done
