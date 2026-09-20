"""Episode-wise v3 cleaning with bounded frame/metadata batches.

No pd.read_parquet / full dataset concat. A bounded episode may span arbitrary
input batches and files. Output file row boundaries and video offsets are kept.
Resume reuses a completed numeric phase and completed video copies; interruption
within the numeric phase restarts that phase (not individual episodes).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
    for folder in ["data", "meta", "videos"]:
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                stat = path.stat()
                result.update(
                    f"{path.relative_to(root).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode()
                )
    return result.hexdigest()


def layout(root, config):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Streaming requires LeRobot v3.0")
    if not 0 < info["total_frames"] <= config.max_frames:
        raise ValueError(f"Empty dataset or total_frames exceeds max_frames={config.max_frames}")
    if not np.isfinite(info["fps"]) or info["fps"] <= 0:
        raise ValueError("Invalid fps")
    ep_paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    episode_count = 0
    referenced = set()
    for path in ep_paths:
        safe_path(root, str(path.relative_to(root)))
        with pq.ParquetFile(path) as parquet:
            episode_count += parquet.metadata.num_rows
            if config.data_file_policy == "metadata_referenced":
                for batch in parquet.iter_batches(
                    batch_size=config.metadata_batch_rows,
                    columns=["data/chunk_index", "data/file_index"],
                ):
                    for row in batch.to_pylist():
                        chunk, file = row["data/chunk_index"], row["data/file_index"]
                        if type(chunk) is not int or type(file) is not int or min(chunk, file) < 0:
                            raise ValueError("Invalid episode data file reference")
                        referenced.add(
                            safe_path(
                                root, info["data_path"].format(chunk_index=chunk, file_index=file)
                            )
                        )
    if not episode_count or episode_count != info["total_episodes"]:
        raise ValueError("Episode metadata count mismatch")
    candidates = sorted((root / "data").glob("chunk-*/*.parquet"))
    if config.data_file_policy == "metadata_referenced":
        missing = referenced - {p.resolve() for p in candidates}
        if missing:
            raise ValueError(
                f"Referenced data files missing or outside supported layout: {sorted(map(str, missing))[:10]}"
            )
    selection = {
        "policy": config.data_file_policy,
        "discovered_files": len(candidates),
        "selected_files": [],
        "excluded_files": [],
        "excluded_rows_known": 0,
    }
    files, count = [], 0
    required = {k for k, value in info["features"].items() if value["dtype"] != "video"}
    for path in candidates:
        safe_path(root, str(path.relative_to(root)))
        relative = path.relative_to(root).as_posix()
        if config.data_file_policy == "metadata_referenced" and path.resolve() not in referenced:
            excluded = {
                "path": relative,
                "reason": "not_referenced_by_episode_metadata",
                "rows": None,
            }
            try:
                with pq.ParquetFile(path) as parquet:
                    excluded["rows"] = parquet.metadata.num_rows
                selection["excluded_rows_known"] += excluded["rows"]
            except (OSError, pa.ArrowException) as exc:
                excluded["header_error"] = str(exc)
            selection["excluded_files"].append(excluded)
            continue
        with pq.ParquetFile(path) as parquet:
            rows, schema = parquet.metadata.num_rows, parquet.schema_arrow
        if not rows or not required.issubset(schema.names):
            raise ValueError(f"Empty data file or missing feature columns: {path}")
        files.append((path, count, count + rows, schema))
        selection["selected_files"].append(relative)
        count += rows
    if count != info["total_frames"]:
        raise ValueError(
            f"Parquet footer row count differs from total_frames: selected={count}, "
            f"declared={info['total_frames']}, policy={config.data_file_policy}. "
            "No missing rows are synthesized or references rewritten."
        )
    expected = 0
    with pq.ParquetFile(root / "meta/tasks.parquet") as parquet:
        for batch in parquet.iter_batches(batch_size=config.batch_rows, columns=["task_index"]):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            if not np.array_equal(values, np.arange(expected, expected + len(values))):
                raise ValueError("Task indices must be contiguous")
            expected += len(values)
    if expected != info["total_tasks"]:
        raise ValueError("Task count mismatch")
    return root, info, files, ep_paths, selection


def metadata_rows(paths, config):
    for path in paths:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=config.metadata_batch_rows):
                for row in batch.to_pylist():
                    yield path, row, parquet.schema_arrow


class FrameCursor:
    def __init__(self, files, config):
        self.generator = self.batches(files, config)
        self.batch = None
        self.offset = 0
        self.peak_rows = 0

    @staticmethod
    def batches(files, config):
        for path, _, _, _ in files:
            with pq.ParquetFile(path) as parquet:
                yield from parquet.iter_batches(batch_size=config.batch_rows)

    def take(self, count):
        pieces = []
        while count:
            if self.batch is None or self.offset == self.batch.num_rows:
                self.batch = next(self.generator, None)
                self.offset = 0
                if self.batch is None:
                    raise ValueError("Data ended before episode metadata")
                self.peak_rows = max(self.peak_rows, self.batch.num_rows)
            length = min(count, self.batch.num_rows - self.offset)
            pieces.append(self.batch.slice(self.offset, length))
            self.offset += length
            count -= length
        return pa.Table.from_batches(pieces)

    def close(self):
        self.generator.close()


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


def validate_episode(root, info, files_by_path, row, index, offset, frame):
    length = len(frame)
    if (
        row["episode_index"] != index
        or row["dataset_from_index"] != offset
        or row["dataset_to_index"] != offset + length
    ):
        raise ValueError("Episode indices/ranges are not contiguous")
    path = safe_path(
        root,
        info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        ),
    )
    bounds = files_by_path.get(path)
    if bounds is None or not bounds[0] <= offset < bounds[1]:
        raise ValueError("Episode data file does not contain its first row")
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


def add_videos(root, info, row, videos):
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        prefix = f"videos/{key}/"
        path = safe_path(
            root,
            info["video_path"].format(
                video_key=key,
                chunk_index=int(row[prefix + "chunk_index"]),
                file_index=int(row[prefix + "file_index"]),
            ),
        )
        start, end = row[prefix + "from_timestamp"], row[prefix + "to_timestamp"]
        if (
            not np.isfinite([start, end]).all()
            or start < 0
            or not np.isclose(end - start, row["length"] / info["fps"], atol=1e-3, rtol=0)
        ):
            raise ValueError(f"Invalid video time interval: episode {row['episode_index']}, {key}")
        relative = path.relative_to(root).as_posix()
        if relative not in videos:
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing/empty referenced video: {path}")
            videos[relative] = {
                "key": key,
                "expected_end": 0.0,
                "intervals": [],
                "episode_indices": [],
            }
        item = videos[relative]
        if item["key"] != key or (item["intervals"] and start < item["expected_end"] - 1e-3):
            raise ValueError(f"Overlapping video intervals or aliased video paths: {relative}")
        item["intervals"].append([start, end])
        item["episode_indices"].append(int(row["episode_index"]))
        item["expected_end"] = end


def transform(frame, info, config, *, cleaning):
    for alias in config.aliases:
        if not np.allclose(
            matrix(frame[alias.source])[:, alias.start : alias.end],
            matrix(frame[alias.target]),
            equal_nan=True,
        ):
            raise ValueError(f"Alias mismatch: {alias.source} -> {alias.target}")
    if not cleaning:
        return {}
    targets = {a.target for a in config.aliases}
    changes = {}
    for key in numeric_keys(info):
        if key in targets:
            continue
        if not info["features"][key]["dtype"].startswith("float"):
            continue  # Do not round-trip large int64 values through float64.
        original = matrix(frame[key])
        values = original.astype(np.float64)
        if not np.isfinite(values).all():
            if config.nonfinite == "error":
                raise ValueError(
                    f"Non-finite values in {key}; episode {frame.episode_index.iloc[0]}"
                )
            for dim in range(values.shape[1]):
                valid = np.isfinite(values[:, dim])
                if not valid.any():
                    raise ValueError(f"Cannot interpolate entirely invalid {key} in this episode")
                x = np.arange(len(values))
                values[~valid, dim] = np.interp(x[~valid], x[valid], values[valid, dim])
        if key in config.bounds:
            values = np.clip(values, *config.bounds[key])
        values = values.astype(info["features"][key]["dtype"])
        count = int((~np.isclose(values, original, rtol=0, atol=0, equal_nan=True)).sum())
        if count:
            put_matrix(frame, key, values, info["features"][key]["dtype"])
            changes[key] = count
    for alias in config.aliases:
        values = matrix(frame[alias.source])[:, alias.start : alias.end]
        count = int(
            (~np.isclose(values, matrix(frame[alias.target]), rtol=0, atol=0, equal_nan=True)).sum()
        )
        if count:
            put_matrix(frame, alias.target, values, info["features"][alias.target]["dtype"])
            changes[alias.target] = count
    return changes


class DataWriter:
    """Keep original parquet file row boundaries, including split episodes."""

    def __init__(self, root, stage, files, config):
        self.root, self.stage, self.files, self.config = root, stage, files, config
        self.index = self.offset = self.pending_rows = 0
        self.writer = None
        self.pending = []

    def flush(self):
        if self.pending:
            self.writer.write_table(
                pa.concat_tables(self.pending), row_group_size=self.config.batch_rows
            )
            self.pending.clear()
            self.pending_rows = 0

    def append(self, table):
        cursor = 0
        while cursor < len(table):
            path, lo, hi, schema = self.files[self.index]
            if self.writer is None:
                dest = self.stage / path.relative_to(self.root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                self.writer = pq.ParquetWriter(dest, schema, compression="zstd")
            count = min(len(table) - cursor, hi - self.offset)
            piece = table.slice(cursor, count).cast(schema)
            self.pending.append(piece)
            self.pending_rows += count
            cursor += count
            self.offset += count
            if self.pending_rows >= self.config.batch_rows or self.offset == hi:
                self.flush()
            if self.offset == hi:
                self.writer.close()
                self.writer = None
                self.index += 1

    def close(self):
        if self.writer is not None:
            self.flush()
            self.writer.close()
            self.writer = None


class MetadataWriter:
    def __init__(self, root, stage, info, config):
        self.root, self.stage, self.info, self.config = root, stage, info, config
        self.path = self.writer = self.schema = None
        self.rows = []

    def flush(self):
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def append(self, path, row, schema):
        if path != self.path:
            self.close()
            numeric = numeric_keys(self.info)
            replaced = {
                f"stats/{key}/{name}"
                for key in numeric
                for name in [
                    "min",
                    "max",
                    "mean",
                    "std",
                    "count",
                    "q01",
                    "q10",
                    "q50",
                    "q90",
                    "q99",
                ]
            }
            fields = [field for field in schema if field.name not in replaced]
            fields += [
                pa.field(name, pa.list_(pa.int64() if name.endswith("/count") else pa.float64()))
                for name in sorted(replaced)
            ]
            self.schema = pa.schema(fields)
            dest = self.stage / path.relative_to(self.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(dest, self.schema, compression="zstd")
            self.path = path
        self.rows.append(row)
        if len(self.rows) >= self.config.metadata_batch_rows:
            self.flush()

    def close(self):
        if self.writer is not None:
            self.flush()
            self.writer.close()
            self.writer = None


def scan(dataset, config, stage=None, observer=None):
    root, info, files, ep_paths, selection = layout(dataset, config)
    check_config(info, config)
    keys = numeric_keys(info)
    dims = {key: int(np.prod(info["features"][key]["shape"])) for key in keys}
    before = {key: NumericSummary(dim) for key, dim in dims.items()}
    stats = (
        {key: OnlineStats(dim, config.quantile_samples) for key, dim in dims.items()}
        if stage
        else {}
    )
    changes, videos = {}, {}
    trajectory_quality = []
    offset = episodes = failed = peak_episode = 0
    files_by_path = {path: (lo, hi) for path, lo, hi, _ in files}
    with ExitStack() as stack:
        reader = FrameCursor(files, config)
        stack.callback(reader.close)
        metadata = metadata_rows(ep_paths, config)
        stack.callback(metadata.close)
        progress = stack.enter_context(
            tqdm(
                total=info["total_frames"],
                unit="frames",
                desc="Cleaning" if stage else "Auditing",
                disable=not config.progress,
            )
        )
        if stage:
            writer = DataWriter(root, stage, files, config)
            meta_writer = MetadataWriter(root, stage, info, config)
            stack.callback(writer.close)
            stack.callback(meta_writer.close)
        for path, row, schema in metadata:
            length = row["length"]
            if not isinstance(length, int) or not 0 < length <= config.max_episode_frames:
                raise ValueError(
                    f"Episode {row['episode_index']} is empty or exceeds max_episode_frames={config.max_episode_frames}"
                )
            peak_episode = max(peak_episode, length)
            raw = reader.take(length)
            frame = raw.to_pandas()
            validate_episode(root, info, files_by_path, row, episodes, offset, frame)
            add_videos(root, info, row, videos)
            if config.quality.enabled:
                from lerobot_cleaner.v30.quality import audit_trajectory
                trajectory_quality.append(audit_trajectory(frame, info["fps"], config.quality))
            if observer is not None:
                observer.process(row, frame)
            if "is_episode_successful" in frame:
                if frame.is_episode_successful.nunique(dropna=False) != 1:
                    raise ValueError("Inconsistent is_episode_successful within an episode")
                failed += int(not bool(frame.is_episode_successful.iloc[0]))
            for key in keys:
                before[key].update(matrix(frame[key]))
            delta = transform(frame, info, config, cleaning=stage is not None)
            if stage:
                for key, count in delta.items():
                    changes[key] = changes.get(key, 0) + count
                for key in keys:
                    values = matrix(frame[key])
                    stats[key].update(values)
                    row.update(
                        {
                            f"stats/{key}/{name}": value
                            for name, value in feature_stats(values).items()
                        }
                    )
                writer.append(
                    pa.Table.from_pandas(frame, schema=raw.schema, preserve_index=False)
                    if delta
                    else raw
                )
                meta_writer.append(path, row, schema)
            offset += length
            episodes += 1
            progress.update(length)
        if offset != info["total_frames"] or episodes != info["total_episodes"]:
            raise ValueError("Unreferenced data rows or missing episodes")
    report = {
        "data_file_selection": selection,
        "dataset": str(root),
        "version": "v3.0",
        "robot_type": info.get("robot_type"),
        "episodes": episodes,
        "frames": offset,
        "fps": info["fps"],
        "tasks": info["total_tasks"],
        "numeric": {key: value.result() for key, value in before.items()},
        "videos": videos,
        "unsuccessful_episodes": failed if "is_episode_successful" in info["features"] else None,
        "video_verification": "metadata_only",
        "engine": "streaming",
        "peak_input_batch_rows": reader.peak_rows,
        "peak_episode_rows": peak_episode,
        "trajectory_quality_input" if stage is not None else "trajectory_quality": trajectory_quality,
        "warning": "Numeric and metadata checks do not verify video pixels, control semantics or task quality.",
    }
    if observer is not None:
        report["dataset_review"] = observer.result()
    if stage:
        original_stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
        original_stats.update({key: value.result() for key, value in stats.items()})
        write_json(stage / "meta/stats.json", original_stats)
        report["changed_values"] = changes
        report["global_quantiles"] = {
            "method": "uniform_reservoir_algorithm_R",
            "samples_per_feature": min(offset, config.quantile_samples),
            "seed": 0,
            "exact_when_all_rows_fit": offset <= config.quantile_samples,
        }
    return report


def verify_videos(root, info, videos, config, *, checkpoint=None, quality=None, preview_root=None):
    """Online frame count/time validation; never store all frame timestamps."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("Video verification requires pip install -e '.[v3-video]'") from exc
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
        with av.open(str(root / relative)) as container:
            stream = container.streams.video[0]
            expected_total = round(item["expected_end"] * info["fps"])
            for frame in tqdm(
                container.decode(stream),
                total=expected_total,
                unit="frames",
                desc=Path(relative).name,
                leave=False,
                disable=not config.progress,
            ):
                if frame.pts is None or [frame.height, frame.width, 3] != shape:
                    raise ValueError(f"Invalid dimensions/PTS: {relative}")
                timestamp = float(frame.pts * stream.time_base)
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
    report = (
        scan(dataset, config, observer=observer) if observer is not None else scan(dataset, config)
    )
    if config.verify_videos:
        root = Path(dataset).resolve()
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        verify_videos(root, info, report["videos"], config)
        report["video_verification"] = "full_decode"
    return report


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


def clean_streaming(
    dataset,
    output,
    config,
    *,
    resume=False,
    observer_factory=None,
    finalize=None,
    job_metadata=None,
    video_quality=None,
):
    root, info, files, _, _ = layout(dataset, config)
    check_config(info, config)
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be a new directory outside the input dataset")
    partial = output.with_name(output.name + ".partial")
    if partial.is_symlink() or partial.resolve().is_relative_to(root):
        raise ValueError("Unsafe partial directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "format": 1,
        "input": str(root),
        "output": str(output),
        "source_fingerprint": fingerprint(root),
        "config": config.model_dump(mode="json"),
    }
    if job_metadata is not None:
        identity["review_settings"] = job_metadata
    exists = partial.exists()
    if exists and not resume:
        raise ValueError(f"Partial job exists: {partial}; use --resume with identical input/config")
    if resume and not exists:
        raise ValueError("No partial job to resume; omit --resume for a new run")
    partial.mkdir(exist_ok=True)
    with job_lock(partial):
        job_file = partial / "job.json"
        if exists:
            if (
                not job_file.is_file()
                or json.loads(job_file.read_text(encoding="utf-8")) != identity
            ):
                raise ValueError(
                    "Resume input/config/source fingerprint mismatch; use a new output directory"
                )
        else:
            write_json(job_file, identity)
        stage = partial / "dataset"
        if stage.is_symlink():
            raise ValueError("Unsafe staged dataset")
        stage.mkdir(exist_ok=True)
        # Conservative capacity estimate: source parquet may compress differently on output.
        video_bytes = sum(p.stat().st_size for p in (root / "videos").rglob("*") if p.is_file())
        other_bytes = sum(
            p.stat().st_size
            for folder in ["data", "meta"]
            for p in (root / folder).rglob("*")
            if p.is_file()
        )
        already_copied = sum(
            p.stat().st_size for p in (stage / "videos").rglob("*.mp4") if p.is_file()
        )
        required = (
            max(0, video_bytes - already_copied)
            + 3 * other_bytes
            + int(config.disk_reserve_gb * 1024**3)
        )
        if shutil.disk_usage(output.parent).free < required:
            raise ValueError(
                f"Insufficient disk space: estimated additional {required / 1024**3:.1f} GiB needed"
            )
        checkpoint = partial / "scan.json"
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            report = saved["report"]
            for relative, checksum in tqdm(
                saved["checksums"].items(), desc="Checking saved data", disable=not config.progress
            ):
                path = safe_path(stage, relative)
                if not path.is_file() or digest(path) != checksum:
                    raise ValueError("Saved data checkpoint is damaged; use a new output directory")
        else:
            # Only remove our own incomplete phase, after validating job identity.
            for name in ["data", "meta"]:
                target = stage / name
                if target.is_symlink() or not target.resolve().is_relative_to(partial.resolve()):
                    raise ValueError("Unsafe partial phase path")
                if target.exists():
                    shutil.rmtree(target)
            (stage / "meta").mkdir()
            for path in (root / "meta").iterdir():
                if path.is_file():
                    shutil.copy2(path, stage / "meta" / path.name)
            report = (
                scan(root, config, stage, observer=observer_factory(root))
                if observer_factory
                else scan(root, config, stage)
            )
            checksums = {
                p.relative_to(stage).as_posix(): digest(p)
                for name in ["data", "meta"]
                for p in (stage / name).rglob("*")
                if p.is_file()
            }
            if fingerprint(root) != identity["source_fingerprint"]:
                raise ValueError("Source changed during scan; do not modify input while cleaning")
            write_json(checkpoint, {"report": report, "checksums": checksums})
        # Older checkpoints skipped input quality during cleaning. Re-audit raw data;
        # never substitute output quality or trust the old empty placeholder.
        if "trajectory_quality_input" not in report:
            report["trajectory_quality_input"] = (
                scan(root, config)["trajectory_quality"] if config.quality.enabled else []
            )
            report.pop("trajectory_quality", None)
            if fingerprint(root) != identity["source_fingerprint"]:
                raise ValueError("Source changed during input quality audit")
            write_json(checkpoint, {"report": report, "checksums": saved["checksums"]})
        copy_checkpoint = partial / "videos.json"
        hashes = (
            json.loads(copy_checkpoint.read_text(encoding="utf-8"))
            if copy_checkpoint.exists()
            else {}
        )
        for relative in tqdm(
            report["videos"], desc="Copying/checksumming videos", disable=not config.progress
        ):
            source, target = safe_path(root, relative), safe_path(stage, relative)
            hashes[relative] = copy_verified(source, target, hashes.get(relative))
            write_json(copy_checkpoint, hashes)
        # Re-read output in bounded batches; this verifies final row/alias/index invariants.
        after_config = config.model_copy(update={"data_file_policy": "strict"})
        audit_config = after_config.model_copy(update={"verify_videos": False})
        after = (
            audit_streaming(stage, audit_config, observer=observer_factory(stage))
            if observer_factory
            else audit_streaming(stage, audit_config)
        )
        effective_quality = dict(video_quality) if video_quality else None
        if effective_quality is not None:
            selected = []
            seen_tasks = set()
            candidates = report.get("dataset_review", {}).get("episode_quality", [])
            limit = effective_quality.pop("preview_limit", 20)
            for row in candidates:
                if row["task_index"] not in seen_tasks and len(selected) < limit:
                    selected.append(row["episode_index"])
                    seen_tasks.add(row["task_index"])
            for row in candidates:
                if row["flags"] and row["episode_index"] not in selected and len(selected) < limit:
                    selected.append(row["episode_index"])
            effective_quality["preview_episodes"] = selected
        if config.verify_videos:
            verify_videos(
                stage,
                info,
                after["videos"],
                config,
                checkpoint=partial / "video_decode.json",
                quality=effective_quality,
            )
            after["video_verification"] = "full_decode"
        if fingerprint(root) != identity["source_fingerprint"]:
            raise ValueError("Source changed during cleaning")
        result = {
            **report,
            "input": str(root),
            "output": str(output),
            "rows_removed": 0,
            "numeric_before": report["numeric"],
            "numeric_after": after["numeric"],
            "trajectory_quality_output": after["trajectory_quality"],
            "videos": after["videos"],
            "video_sha256": hashes,
            "video_verification": after["video_verification"],
            "policy": "Preserve all selected rows/tasks/extra columns/video offsets; no success filtering. "
            "Unreferenced data files are excluded only under the explicit metadata_referenced policy.",
            "resume_policy": "Completed numeric phase + individually verified video copies; incomplete numeric phase restarts.",
        }
        if observer_factory:
            if report["dataset_review"]["tasks"] != after["dataset_review"]["tasks"]:
                raise ValueError("Output language/task counts differ from input")
            result["output_review"] = after["dataset_review"]
        report_dir = stage / "cleaning_report"
        report_dir.mkdir(exist_ok=True)
        write_json(report_dir / "report.json", result)
        write_json(report_dir / "input_audit.json", report)
        write_json(report_dir / "data_file_selection.json", report["data_file_selection"])
        (report_dir / "cleaning_config.used.yaml").write_text(
            yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
        )
        (report_dir / "report.md").write_text(
            f"# Streaming v3 cleaning report\n\nEpisodes: {report['episodes']}; frames: {report['frames']}.\n\n"
            f"Changed values: {report['changed_values']}. No frames removed from the selected dataset.\n\n"
            f"Data files: {len(report['data_file_selection']['selected_files'])} selected, "
            f"{len(report['data_file_selection']['excluded_files'])} excluded by explicit policy. "
            "See data_file_selection.json; original files are untouched.\n\n"
            f"Video verification: {after['video_verification']}; copies verified by SHA256.\n\n"
            f"Peak input batch: {report['peak_input_batch_rows']} rows; peak episode: {report['peak_episode_rows']} rows.\n\n"
            f"Global quantiles: uniform reservoir ({min(report['frames'], config.quantile_samples)} rows/feature, seed 0); "
            "episode quantiles exact. Mean/std/min/max use all rows (floating-point arithmetic).\n\n"
            "This is not a LingBot training/runtime validation. Recompute LingBot normalization separately.\n",
            encoding="utf-8",
        )
        if finalize is not None:
            finalize(stage, result)
        write_json(
            report_dir / "COMPLETE.json",
            {
                "status": "complete",
                "input": str(root),
                "output": str(output),
                "report_sha256": digest(report_dir / "report.json"),
            },
        )
        if output.exists():
            raise ValueError("Output appeared during processing; refusing to overwrite")
        stage.rename(output)
    # Keep the small checkpoint folder for provenance; final output never lives inside it.
    return result
