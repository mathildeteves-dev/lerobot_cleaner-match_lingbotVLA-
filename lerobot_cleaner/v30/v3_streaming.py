"""Episode-wise quality/cleaning over the official LeRobot storage boundary.

Official initialization may cache the full dataset. Each quality evaluation
receives one episode; cleaner write buffers are bounded. Output file row boundaries and video offsets are kept.
Resume reuses a completed numeric phase and completed video copies; interruption
within the numeric phase restarts that phase (not individual episodes).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from lerobot_cleaner.v30.v3 import feature_stats, matrix, numeric_keys, put_matrix, safe_path
from lerobot_cleaner.v30.v3_stream_stats import NumericSummary, OnlineStats


def digest(path):
    with Path(path).open("rb") as handle:
        result = hashlib.sha256()
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(root):
    """Identity check, not a substitute for full source content hashes.

    Inputs must remain immutable; path/size/mtime changes invalidate resume.
    """
    result = hashlib.sha256((root / "meta/info.json").read_bytes())
    for folder in ["data", "meta", "videos", "images", "depth"]:
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                stat = path.stat()
                result.update(
                    f"{path.relative_to(root).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
                )
    return result.hexdigest()


def layout(root, config):
    """Compatibility view of the storage layer's writer contract."""
    from lerobot_cleaner.storage import OfficialStorage
    with OfficialStorage(root, config) as storage:
        files, metadata, selection = storage.output_layout()
        return storage.root, storage.info, files, [p for p, *_ in metadata], selection


def check_config(info, config):
    keys = numeric_keys(info)
    targets = {a.target for a in config.aliases}
    if len(targets) != len(config.aliases) or targets.intersection(config.bounds):
        raise ValueError("Duplicate alias targets or bounds on alias targets")
    for alias in config.aliases:
        if alias.source not in keys or alias.target not in keys or alias.source in targets:
            raise ValueError("Aliases require numeric columns without chains/cycles")
        source = int(np.prod(info["features"][alias.source]["shape"]))
        target = int(np.prod(info["features"][alias.target]["shape"]))
        if not 0 <= alias.start < alias.end <= source or alias.end - alias.start != target:
            raise ValueError("Invalid alias slice")
    for key, bounds in config.bounds.items():
        if key not in keys or not info["features"][key]["dtype"].startswith("float"):
            raise ValueError(f"Bounds require a floating-point feature: {key}")
        if key in {"timestamp", "index", "frame_index", "episode_index", "task_index"}:
            raise ValueError("Bounds on indexing/time fields are not supported")
        if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid bounds: {key}")


def validate_episode(info, row, index, offset, frame):
    length = len(frame)
    if (
        row["episode_index"] != index
        or row["dataset_from_index"] != offset
        or row["dataset_to_index"] != offset + length
    ):
        raise ValueError("Episode indices/ranges are not contiguous")
    for key, expected in [
        ("index", np.arange(offset, offset + length)),
        ("episode_index", np.full(length, index)),
        ("frame_index", np.arange(length)),
    ]:
        if not np.array_equal(frame[key], expected):
            raise ValueError(f"Invalid {key} in episode {index}")
    if not np.allclose(frame.timestamp, np.arange(length) / info["fps"], atol=1e-4, rtol=0):
        raise ValueError(f"Non-uniform timestamps in episode {index}")
    tasks = frame.task_index.to_numpy()
    if (
        not np.isfinite(tasks).all()
        or not ((tasks >= 0) & (tasks < info["total_tasks"]) & (tasks == np.floor(tasks))).all()
    ):
        raise ValueError(f"Undefined task indices (unresolved language) in episode {index}")
    for key in numeric_keys(info):
        if matrix(frame[key]).shape[1] != int(np.prod(info["features"][key]["shape"])):
            raise ValueError(f"Feature dimension mismatch: {key}")


def scan(dataset, config, stage=None, observer=None):
    """Read-only compatibility entry. Mutation requires clean_v3 and its plans."""
    if stage is not None:
        raise ValueError("scan is read-only; use clean_v3 to execute transform plans")
    from .pipeline import audit_pipeline
    return audit_pipeline(dataset, config, observer=observer)


def verify_videos(root, info, videos, config, *, checkpoint=None, quality=None, preview_root=None):
    """Online frame count/time validation; never store all frame timestamps."""
    from lerobot_cleaner.storage.video import decoded_video
    from lerobot_cleaner.v30.video_review import VideoReview

    saved = (
        json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint and checkpoint.exists()
        else {}
    )
    artifact_root = preview_root or root
    for relative, item in tqdm(videos.items(), desc="Decoding videos", disable=not config.progress):
        identity = None
        if checkpoint:
            identity = {
                "sha256": digest(root / relative),
                "fps": info["fps"],
                "shape": info["features"][item["key"]]["shape"],
                "intervals": item["intervals"],
                "episode_indices": item.get("episode_indices"),
                "quality": quality,
                "version": 1,
            }
            cached = saved.get(relative, {})
            if cached.get("identity") == identity and all(
                safe_path(artifact_root, name).is_file()
                and digest(safe_path(artifact_root, name)) == checksum
                for name, checksum in cached.get("preview_hashes", {}).items()
            ):
                item.update(cached["result"])
                item["decode_reused"] = True
                continue
        intervals = item["intervals"]
        counts = np.zeros(len(intervals), dtype=np.int64)
        interval = frames = 0
        previous = None
        shape = info["features"][item["key"]]["shape"]
        review = VideoReview(artifact_root, relative, item, info, quality) if quality else None
        with decoded_video(root / relative) as decoded_frames:
            expected_total = round(item["expected_end"] * info["fps"])
            for frame in tqdm(
                decoded_frames,
                total=expected_total,
                unit="frames",
                desc=Path(relative).name,
                leave=False,
                disable=not config.progress,
            ):
                if frame.pts is None or [frame.height, frame.width, 3] != shape:
                    raise ValueError(f"Invalid dimensions/PTS: {relative}")
                timestamp = float(frame.pts * frame.time_base)
                if previous is not None and not np.isclose(
                    timestamp - previous, 1 / info["fps"], atol=1e-3, rtol=0
                ):
                    raise ValueError(f"Non-uniform video timestamps: {relative}")
                previous = timestamp
                while interval < len(intervals) and timestamp >= intervals[interval][1] - 1e-4:
                    interval += 1
                if interval < len(intervals) and timestamp >= intervals[interval][0] - 1e-4:
                    counts[interval] += 1
                    if review:
                        review.sample(frame, interval, timestamp, int(counts[interval]))
                frames += 1
        expected = [round((end - start) * info["fps"]) for start, end in intervals]
        if not frames or not np.array_equal(counts, expected):
            raise ValueError(f"Video interval frame count mismatch: {relative}")
        item["decoded_frames"] = frames
        item["interval_frame_counts"] = counts.tolist()
        item["decode_reused"] = False
        if review:
            item["visual_review"] = review.result()
        if checkpoint:
            fields = ["decoded_frames", "interval_frame_counts", "decode_reused", "visual_review"]
            previews = review.preview_paths if review else []
            saved[relative] = {
                "identity": identity,
                "result": {key: item[key] for key in fields if key in item},
                "preview_hashes": {p: digest(safe_path(artifact_root, p)) for p in previews},
            }
            write_json(checkpoint, saved)


def audit_streaming(dataset, config, *, observer=None):
    from .pipeline import audit_pipeline
    return audit_pipeline(dataset, config, observer=observer)


@contextmanager
def job_lock(folder):
    lock = folder / ".lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(
                "Do not run concurrent jobs. Remove only after confirming the old process has stopped.\n"
            )
    except FileExistsError as exc:
        raise ValueError(
            f"Job locked: {lock}. Confirm previous process stopped before removing this lock."
        ) from exc
    try:
        yield
    finally:
        lock.unlink()


def copy_verified(source, target, expected=None):
    if expected and target.is_file() and digest(target) == expected:
        return expected
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    shutil.copy2(source, temporary)
    source_hash = digest(source)
    if digest(temporary) != source_hash:
        raise RuntimeError(f"Copy checksum mismatch: {source}")
    temporary.replace(target)
    return source_hash


def clean_streaming(dataset, output, config, *, resume=False, observer_factory=None,
                    finalize=None, job_metadata=None, video_quality=None):
    from .pipeline import clean_pipeline
    return clean_pipeline(dataset, output, config, resume=resume,
                          observer_factory=observer_factory, finalize=finalize,
                          job_metadata=job_metadata, video_quality=video_quality)
