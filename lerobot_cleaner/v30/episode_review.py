"""Episode diagnostics observed during the core bounded-memory data scan."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot_cleaner.v30.quality import TrajectoryQualityConfig, audit_trajectory
from lerobot_cleaner.v30.review_profile import load_profile
from lerobot_cleaner.v30.v3 import matrix, numeric_keys
from lerobot_cleaner.adapters.language import task_catalog, resolve_language
from lerobot_cleaner.core.language import text_error, valid_index


def task_texts(frame):
    """Support both an explicit task column and LeRobot's string pandas index."""
    if "task_index" not in frame:
        raise ValueError("Missing task_index in tasks.parquet")
    ids = frame["task_index"].to_numpy()
    if not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("Task indices must be contiguous integers starting at zero")
    values = frame["task"].tolist() if "task" in frame else frame.index.tolist()
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Missing/blank task text; expected task column or string index")
    return dict(zip(map(int, ids), values))


def longest_run(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


class DatasetReview:
    def __init__(self, root, profile=None):
        self.root = Path(root).resolve()
        self.profile = profile or load_profile()
        from lerobot_cleaner.storage import OfficialStorage
        with OfficialStorage(self.root) as storage:
            self.info = storage.info
            self.task_catalog = task_catalog(storage.tasks)
            counts = Counter(index for index, _ in self.task_catalog if valid_index(index))
            self.texts = {int(index): text for index, text in self.task_catalog
                          if valid_index(index) and counts[index] == 1 and text_error(text) is None}
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("Review requires LeRobot v3.0")
        features = self.info["features"]
        for key, width in self.profile.features.items():
            feature = features.get(key, {})
            if feature.get("shape") != [width] or not feature.get("dtype", "").startswith("float"):
                raise ValueError(
                    f"Not the supported LIBERO schema/profile: {key} must be float[{width}]"
                )
        cameras = {k for k, v in features.items() if v["dtype"] in {"video", "image"}}
        if cameras != set(self.profile.cameras):
            raise ValueError(
                f"Expected profile cameras {self.profile.cameras}; got {sorted(cameras)}"
            )
        self.frames_without_task_text = 0
        # Generic language findings report invalid tables; profile summaries must
        # not prevent the unified quality report from being written.
        self.feature_schema = None
        self.rows, self.hashes = [], {}
        self.frame_counts, self.episode_counts = Counter(), Counter()

    def set_feature_schema(self, schema):
        """Use the same canonical layout as the pipeline quality checks."""
        self.feature_schema = schema

    def process(self, metadata, frame):
        eid, length = int(metadata["episode_index"]), len(frame)
        language = resolve_language(frame, eid, metadata, self.task_catalog)
        tasks = Counter(int(item.task_index) for item in language.samples
                        if valid_index(item.task_index) and item.task_index in self.texts)
        self.frames_without_task_text += length - sum(tasks.values())
        self.frame_counts.update(tasks)
        self.episode_counts.update(tasks.keys())
        task = next(iter(tasks)) if len(tasks) == 1 else None
        for alias in self.profile.aliases:
            if not np.allclose(
                matrix(frame[alias.source])[:, alias.start : alias.end],
                matrix(frame[alias.target]),
                rtol=self.profile.alias_rtol,
                atol=self.profile.alias_atol,
                equal_nan=True,
            ):
                raise ValueError(
                    f"State alias mismatch in episode {eid}: {alias.source} -> {alias.target}"
                )
        nonfinite = {
            k: int((~np.isfinite(matrix(frame[k]))).sum()) for k in numeric_keys(self.info)
        }
        nonfinite = {k: n for k, n in nonfinite.items() if n}
        row = {
            "episode_index": eid,
            "task_index": task,
            "length": length,
            "seconds": length / self.info["fps"],
            "flags": [],
            "nonfinite": nonfinite,
            "task": self.texts.get(task),
            "task_indices": sorted(tasks),
        }
        trajectory_quality = audit_trajectory(frame, self.info["fps"], TrajectoryQualityConfig(
            state_column=self.profile.state_feature, action_column=self.profile.action_feature,
            groups=self.profile.quality.groups,
            joint_static_ratio=self.profile.quality.joint_static_ratio,
        ), schema=self.feature_schema)
        row["trajectory_checks"] = trajectory_quality["checks"]
        if "groups" in trajectory_quality:
            row["trajectory_groups"] = trajectory_quality["groups"]
        if nonfinite:
            row["flags"].append("nonfinite")
        else:
            state, action = (
                matrix(frame[self.profile.state_feature]),
                matrix(frame[self.profile.action_feature]),
            )
            settings, fps = self.profile.quality, self.info["fps"]
            still = np.max(np.abs(np.diff(state, axis=0)), axis=1) <= settings.static_epsilon
            row["static_transition_fraction"] = float(still.mean()) if len(still) else 0.0
            row["longest_static_seconds"] = longest_run(still) / fps
            motion = (
                action
                if settings.motion_action_dims is None
                else action[:, settings.motion_action_dims]
            )
            delta = np.diff(motion, axis=0)
            jump = np.max(np.abs(delta), axis=1)
            indices = np.flatnonzero(jump > settings.action_jump) + 1
            row["action_jump_count"] = len(indices)
            row["action_jump_frames_first_20"] = indices[:20].tolist()
            row["max_action_step"] = float(jump.max()) if len(jump) else 0.0
            jitter = (
                (delta[1:] * delta[:-1] < 0)
                & (np.abs(delta[1:]) > settings.jitter_step)
                & (np.abs(delta[:-1]) > settings.jitter_step)
            ).any(axis=1)
            row["jitter_transition_fraction"] = float(jitter.mean()) if len(jitter) else 0.0
            if row["longest_static_seconds"] >= settings.static_seconds:
                row["flags"].append("long_static_state")
            if len(indices):
                row["flags"].append("action_step_candidate")
            if row["jitter_transition_fraction"] > settings.jitter_fraction:
                row["flags"].append("action_jitter_candidate")
            if not settings.min_seconds <= row["seconds"] <= settings.max_seconds:
                row["flags"].append("duration_candidate")
            digest = hashlib.sha256()
            for array in (state, action):
                digest.update(str(array.shape).encode())
                digest.update(np.asarray(array, dtype="<f8").tobytes())
            key = digest.hexdigest()
            if key in self.hashes:
                row["duplicate_numeric_trajectory_of"] = self.hashes[key]
                row["flags"].append("exact_duplicate_state_action")
            else:
                self.hashes[key] = eid
        self.rows.append(row)

    def result(self):
        lengths = [r["length"] for r in self.rows]
        return {
            "input": str(self.root),
            "profile": self.profile.name,
            "episodes": len(lengths),
            "frames": sum(lengths),
            "fps": self.info["fps"],
            "frames_without_task_text": self.frames_without_task_text,
            "language_source": "meta/tasks.parquet joined via task_index",
            "tasks": [
                {
                    "task_index": k,
                    "text": text,
                    "episodes": self.episode_counts[k],
                    "frames": self.frame_counts[k],
                }
                for k, text in self.texts.items()
            ],
            "success_labels_available": "is_episode_successful" in self.info["features"],
            "episode_lengths": {
                "min": min(lengths),
                "max": max(lengths),
                "median": float(np.median(lengths)),
            },
            "quality_candidates": sum(bool(row["flags"]) for row in self.rows),
            "episode_quality": self.rows,
            "action_transform": "none",
            "semantics": self.profile.semantics.model_dump(),
            "training_runtime_validated": False,
            "warning": "Quality flags are native-unit heuristics, not proof of failure. No automatic deletion.",
        }
