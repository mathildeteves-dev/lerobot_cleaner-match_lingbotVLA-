"""LeRobot 0.4.2 storage backend; raw tables, without training transforms.

The official dataset owns schema interpretation and episode boundaries. We use
its Hugging Face table rather than __getitem__, which decodes video and adds
training-only fields. Output layout and finalization remain cleaner concerns.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def open_official_dataset(root):
    root = Path(root).resolve()
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise ValueError("Missing local meta/info.json")
    if json.loads(info_path.read_text(encoding="utf-8")).get("codebase_version") != "v3.0":
        raise ValueError("LeRobot 0.4.2 backend requires v3.0; use run or export-lingbot for v2.1")
    for name in ("info.json", "stats.json", "tasks.parquet"):
        if not (root / "meta" / name).is_file():
            raise ValueError(f"Incomplete local LeRobot dataset: meta/{name}")
    if not any((root / "meta/episodes").glob("chunk-*/*.parquet")):
        raise ValueError("Missing local episode metadata")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "reader_backend=lerobot requires LeRobot 0.4.2; install '.[lingbot]' "
            "in the target environment. There is no native fallback."
        ) from exc

    class LocalDataset(LeRobotDataset):
        def download(self, *args, **kwargs):
            raise RuntimeError("Incomplete local dataset; automatic download is disabled")

        def load_hf_dataset(self):
            try:
                return super().load_hf_dataset()
            except (AssertionError, FileNotFoundError, NotADirectoryError) as exc:
                raise RuntimeError("Official loader rejected local data") from exc

    # Non-version revision avoids remote version negotiation on missing data.
    return LocalDataset(repo_id="local/cleaner", root=root, revision="local",
                        download_videos=False, video_backend="pyav")


class OfficialReader:
    def __init__(self, root, *, max_episode_frames=100_000):
        self.root = Path(root).resolve()
        self.dataset = open_official_dataset(self.root)
        self.table = self.dataset.hf_dataset.with_format("arrow")
        self.info = self.dataset.meta.info
        self.max_episode_frames = max_episode_frames
        self.offset = 0
        self.peak_rows = 0
        if len(self.table) != self.info["total_frames"]:
            raise ValueError("Official dataset row count differs from metadata")

    def rows(self, start, stop):
        if not 0 <= start < stop <= len(self.table):
            raise ValueError("Invalid official dataset row interval")
        table = self.table[start:stop]
        if not np.array_equal(table["index"].to_numpy(), np.arange(start, stop)):
            raise ValueError("Official dataset order/global indices differ from storage order")
        self.peak_rows = max(self.peak_rows, stop - start)
        return table

    def episode(self, episode_id):
        if not 0 <= episode_id < self.info["total_episodes"]:
            raise KeyError(f"Unknown episode: {episode_id}")
        row = dict(self.dataset.meta.episodes[episode_id])
        start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if not 0 < stop - start <= self.max_episode_frames:
            raise ValueError("Episode exceeds configured frame limit")
        table = self.rows(start, stop)
        if row["length"] != len(table) or not np.all(table["episode_index"].to_numpy() == episode_id):
            raise ValueError("Official episode metadata does not match data")
        return table, row

    def take(self, count):
        table = self.rows(self.offset, self.offset + count)
        self.offset += count
        return table

    def close(self):
        # No finalize(): this is a reader, never a dataset writer.
        self.table = self.dataset = None


class OfficialStorage(OfficialReader):
    """Storage facade around official data, metadata and path abstractions.

    Direct parquet footer inspection exists only for lossless output layout and
    schema preservation. Episode rows and offsets always come from official meta;
    input sample values always come from official hf_dataset.
    """
    def __init__(self, root, config=None):
        from .prepare import prepare_dataset
        self.source_root = Path(root).resolve()
        super().__init__(prepare_dataset(root, config),
                         max_episode_frames=getattr(config, "max_episode_frames", 100_000))
        self.config = config
        self.meta = self.dataset.meta
        self.stats = self.meta.stats
        self.tasks = self.meta.tasks
        task_ids = self.tasks["task_index"].to_numpy()
        if (len(task_ids) != self.info["total_tasks"] or
                not np.array_equal(np.sort(task_ids), np.arange(len(task_ids)))):
            self.close()
            raise ValueError("Official task indices/count are inconsistent")
        self._layout = None
        limit = getattr(config, "max_frames", None)
        if limit is not None and not 0 < self.info["total_frames"] <= limit:
            self.close()
            raise ValueError("Empty dataset or total_frames exceeds max_frames")
        if not np.isfinite(self.info["fps"]) or self.info["fps"] <= 0:
            self.close()
            raise ValueError("Invalid fps")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Dataset path escapes root: {relative}")
        return path

    def episode_metadata(self, index):
        row = dict(self.meta.episodes[index])
        if int(row["episode_index"]) != index:
            raise ValueError("Official episode metadata is not in index order")
        return row

    def iter_episode_metadata(self):
        for index in range(self.info["total_episodes"]):
            yield self.episode_metadata(index)

    def data_path(self, episode_index):
        return self.path(self.meta.get_data_file_path(episode_index))

    def video_references(self, episode_index):
        row = self.episode_metadata(episode_index)
        for key in self.meta.video_keys:
            prefix = f"videos/{key}/"
            yield {"key": key,
                   "path": self.path(self.meta.get_video_file_path(episode_index, key)),
                   "start": float(row[prefix + "from_timestamp"]),
                   "end": float(row[prefix + "to_timestamp"])}

    def video_frames(self, episode_index, key, timestamps=None):
        """Use the official decoder and offsets, without applying training transforms."""
        from lerobot.datasets.video_utils import decode_video_frames
        row = self.episode_metadata(episode_index)
        if key not in self.meta.video_keys:
            raise KeyError(key)
        if timestamps is None:
            timestamps = np.arange(int(row["length"])) / self.info["fps"]
        offset = float(row[f"videos/{key}/from_timestamp"])
        query = (np.asarray(timestamps, dtype=float) + offset).tolist()
        return decode_video_frames(
            self.path(self.meta.get_video_file_path(episode_index, key)), query,
            self.dataset.tolerance_s, self.dataset.video_backend)

    def add_videos(self, row, videos):
        for reference in self.video_references(int(row["episode_index"])):
            key, path, start, end = (reference[name] for name in ("key", "path", "start", "end"))
            if (not np.isfinite([start, end]).all() or start < 0 or
                    not np.isclose(end - start, row["length"] / self.info["fps"], atol=1e-3, rtol=0)):
                raise ValueError(f"Invalid video interval: {key}")
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing/empty referenced video: {path}")
            relative = path.relative_to(self.root).as_posix()
            item = videos.setdefault(relative, {"key": key, "expected_end": 0.,
                                               "intervals": [], "episode_indices": []})
            if item["key"] != key or (item["intervals"] and start < item["expected_end"] - 1e-3):
                raise ValueError(f"Overlapping video intervals: {relative}")
            item["intervals"].append([start, end])
            item["episode_indices"].append(int(row["episode_index"]))
            item["expected_end"] = end

    def output_layout(self):
        """Physical writer contract; never used to infer episode boundaries."""
        if self._layout is not None:
            return self._layout
        files, offset = [], 0
        required = {k for k, v in self.info["features"].items() if v["dtype"] != "video"}
        # Expand the official metadata templates only inside the storage layer.
        pattern = self.info["data_path"].replace("{chunk_index:03d}", "*").replace("{file_index:03d}", "*")
        import re
        pattern = re.sub(r"\{[^}]+\}", "*", pattern)
        for candidate in sorted(self.root.glob(pattern)):
            path = self.path(candidate.relative_to(self.root))
            with pq.ParquetFile(path) as parquet:
                count, schema = parquet.metadata.num_rows, parquet.schema_arrow
            if count <= 0 or not required.issubset(schema.names):
                raise ValueError(f"Empty shard or missing source columns: {path}")
            files.append((path, offset, offset + count, schema))
            offset += count
        if offset != self.info["total_frames"]:
            raise ValueError("Source shard row count differs from official metadata")
        metadata_files, count = [], 0
        for candidate in sorted((self.root / "meta/episodes").glob("chunk-*/*.parquet")):
            path = self.path(candidate.relative_to(self.root))
            with pq.ParquetFile(path) as parquet:
                length, schema = parquet.metadata.num_rows, parquet.schema_arrow
            metadata_files.append((path, count, count + length, schema))
            count += length
        if count != self.info["total_episodes"]:
            raise ValueError("Metadata shard row count differs from official metadata")
        selection = {"policy": "strict", "discovered_files": len(files),
                     "selected_files": [p.relative_to(self.root).as_posix() for p, *_ in files],
                     "excluded_files": [], "excluded_rows_known": 0}
        self._layout = (files, metadata_files, selection)
        return self._layout

    def metadata_rows(self):
        _, shards, _ = self.output_layout()
        for path, start, stop, schema in shards:
            for index in range(start, stop):
                yield path, self.episode_metadata(index), schema

    def storage_schema(self, offset):
        files, _, _ = self.output_layout()
        return next(schema for _, start, stop, schema in files if start <= offset < stop)

    def restore_schema(self, table, offset):
        schema = self.storage_schema(offset)
        if set(table.column_names) != set(schema.names):
            raise ValueError("Official loader dropped or added storage columns")
        return table.select(schema.names).cast(schema)

    def validate_episode(self, row, offset):
        index = int(row["episode_index"])
        if int(row["dataset_from_index"]) != offset:
            raise ValueError("Official episode ranges are not contiguous")
        path = self.data_path(index)
        files, _, _ = self.output_layout()
        if not any(p == path and start <= offset < stop for p, start, stop, _ in files):
            raise ValueError("Official episode data reference does not contain its first row")

    def memory_snapshot(self):
        files, metadata_files, _ = self.output_layout()
        parts = [(p, self.restore_schema(self.rows(start, stop), start).to_pandas())
                 for p, start, stop, _ in files]
        episodes = list(self.iter_episode_metadata())
        ep_tables = [(p, pd.DataFrame(episodes[start:stop])) for p, start, stop, _ in metadata_files]
        videos = {}
        for row in episodes:
            self.add_videos(row, videos)
        return (self.root, self.info, pd.concat([table for _, table in parts], ignore_index=True),
                pd.DataFrame(episodes), parts, ep_tables, videos)

    def close(self):
        super().close()
        self.meta = None
