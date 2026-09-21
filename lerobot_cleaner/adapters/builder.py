"""Episode assembly, separate from source metadata and storage writers.

Already aligned parquet rows are preserved by default. Asynchronous streams
require an explicit reference timeline and matching policy; no video is decoded.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .episode import EpisodeRef, UnifiedEpisode
from .schema import vector_values


@dataclass(frozen=True)
class AssemblyPolicy:
    alignment: str = "exact"  # exact, nearest, backward
    tolerance: float | None = None
    missing: str = "preserve"  # preserve, error, zero, forward_fill
    pad_to: int | None = None
    padding: str = "nan"  # nan, zero, edge

    def __post_init__(self):
        if self.alignment not in {"exact", "nearest", "backward"}:
            raise ValueError("Unknown alignment policy")
        if self.alignment != "exact" and self.tolerance is None:
            raise ValueError("Time matching requires an explicit tolerance")
        if self.tolerance is not None and (not np.isfinite(self.tolerance) or self.tolerance < 0):
            raise ValueError("tolerance must be finite and nonnegative")
        if self.missing not in {"preserve", "error", "zero", "forward_fill"}:
            raise ValueError("Unknown missing value policy")
        if self.padding not in {"nan", "zero", "edge"}:
            raise ValueError("Unknown padding policy")
        if self.pad_to is not None and (type(self.pad_to) is not int or self.pad_to < 1):
            raise ValueError("pad_to must be a positive integer")


class EpisodeBuilder:
    def __init__(self, schema, fps, policy=None):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.schema = schema
        self.fps = fps
        self.policy = policy or AssemblyPolicy()

    def build(self, frame, *, ref=None, metadata=None):
        """Wrap synchronized rows without repairing evidence needed by checks."""
        if len(frame):
            if "episode_index" not in frame or frame.episode_index.isna().any() or frame.episode_index.nunique() != 1:
                raise ValueError("Expected one nonempty episode")
            episode_id = int(frame.episode_index.iloc[0])
            if ref is not None and ref.episode_index != episode_id:
                raise ValueError("Episode reference does not match frame")
            ref = ref or EpisodeRef(episode_id, len(frame), [], None)
        elif ref is None:
            raise ValueError("Empty episodes require an episode reference")
        if ref.length != len(frame):
            raise ValueError("Episode reference length does not match frame")
        return UnifiedEpisode(
            ref=ref, df=frame, keep_indices=list(range(len(frame))), fps=self.fps,
            state_features=None if self.schema.raw_vectors else self.schema.states,
            action_features=None if self.schema.raw_vectors else self.schema.actions,
            camera_features=self.schema.cameras, feature_schema=self.schema,
            assembly_policy=self.policy, metadata=dict(metadata or {}))

    def build_streams(self, streams, timeline, *, episode_id, key="timestamp", metadata=None):
        """Pair source columns on one timeline before feature extraction.

        streams maps a source column to a DataFrame containing that column and
        timestamp (seconds) or frame_index. Nearest ties choose the earlier row.
        Missing samples stay NaN until the configured missing-value policy runs.
        Padding is appended only, with an explicit mask and no synthetic row IDs.
        """
        if key not in {"timestamp", "frame_index"}:
            raise ValueError("Alignment key must be timestamp or frame_index")
        timeline = np.asarray(timeline, dtype=float)
        self._validate_timeline(timeline)
        if key == "frame_index" and np.any(timeline != np.floor(timeline)):
            raise ValueError("frame_index must contain integers")
        count = len(timeline)
        if not count:
            raise ValueError("Reference timeline must not be empty")
        columns = {}
        for feature in self.schema.states + self.schema.actions:
            for part in feature.slices:
                columns[part.column] = max(columns.get(part.column, 0), part.end)
        frame = pd.DataFrame({key: timeline, "episode_index": episode_id})
        matched = {}
        for column, width in columns.items():
            stream = streams.get(column)
            values = np.full((count, width), np.nan)
            mask = np.zeros(count, dtype=bool)
            if stream is not None:
                if key not in stream or column not in stream:
                    raise ValueError(f"Stream requires {key} and {column}")
                if "episode_index" in stream and not stream.episode_index.eq(episode_id).all():
                    raise ValueError("Cannot align samples from another episode")
                source_time = stream[key].to_numpy(dtype=float)
                self._validate_timeline(source_time)
                if key == "frame_index" and np.any(source_time != np.floor(source_time)):
                    raise ValueError("frame_index must contain integers")
                if len(stream):
                    source = vector_values(stream[column], width)
                    if source.ndim == 1:
                        source = source[:, None]
                    if source.ndim != 2 or source.shape[1] < width:
                        raise ValueError(f"Invalid source width: {column}")
                    values = np.full((count, source.shape[1]), np.nan)
                    right = pd.DataFrame({key: source_time, "sample": np.arange(len(stream))})
                    left = pd.DataFrame({key: timeline})
                    if self.policy.alignment == "exact":
                        selected = left.merge(right, on=key, how="left")["sample"]
                    else:
                        selected = pd.merge_asof(left, right, on=key,
                                                 direction=self.policy.alignment,
                                                 tolerance=self.policy.tolerance)["sample"]
                    mask = selected.notna().to_numpy()
                    values[mask] = source[selected[mask].to_numpy(dtype=int)]
            frame[column] = list(values)
            matched[column] = mask.tolist()
        if key == "frame_index":
            frame["timestamp"] = timeline / self.fps
        length = max(count, self.policy.pad_to or count)
        if length > count:
            extra = pd.DataFrame({"episode_index": [episode_id] * (length - count),
                                  "timestamp": frame.timestamp.iloc[-1] + np.arange(1, length-count+1) / self.fps})
            for column in columns:
                last = frame[column].iloc[-1]
                value = (last.copy() if self.policy.padding == "edge" else
                         np.full(len(last), 0. if self.policy.padding == "zero" else np.nan))
                extra[column] = [value.copy() for _ in range(length-count)]
            frame = pd.concat([frame, extra], ignore_index=True)
        meta = {**(metadata or {}), "assembly": {"key": key, "matched": matched,
                "valid_mask": [True] * count + [False] * (length-count), "source_rows": count}}
        return self.build(frame, metadata=meta)

    @staticmethod
    def _validate_timeline(values):
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError("Alignment timelines must be finite and strictly increasing")

    def arrays(self, frame):
        def extract(features, column):
            if self.schema.raw_vectors:
                if column not in frame or not len(frame):
                    return np.empty((len(frame), 0))
                values = np.stack(frame[column].to_numpy()).astype(float)
                values = values[:, None] if values.ndim == 1 else values
            else:
                parts = [feature.extract(frame) for feature in features] if len(frame) else []
                values = np.concatenate(parts, axis=1) if parts else np.empty((len(frame), sum(f.width for f in features)))
            missing = ~np.isfinite(values)
            if missing.any():
                if self.policy.missing == "error":
                    raise ValueError(f"Missing/nonfinite values in {column}")
                if self.policy.missing == "zero":
                    values = np.where(missing, 0., values)
                if self.policy.missing == "forward_fill":
                    values = pd.DataFrame(np.where(missing, np.nan, values)).ffill().to_numpy()
            return values
        return (extract(self.schema.states, self.schema.state_column),
                extract(self.schema.actions, self.schema.action_column))


# Alternative terminology for the same assembly layer.
EpisodeAssembler = EpisodeBuilder
EpisodeAdapter = EpisodeBuilder
