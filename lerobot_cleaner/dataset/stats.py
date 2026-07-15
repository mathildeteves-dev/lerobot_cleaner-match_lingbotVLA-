"""Streaming statistics over the global concatenation of all cleaned frames.

GR00T reads ``meta/stats.json`` and slices it by modality.json start/end at load
time (see Isaac-GR00T lerobot_episode_loader.get_dataset_statistics). Each feature
needs mean/std/min/max/q01/q99 over the *entire* dataset.

We compute mean/std/min/max exactly via running accumulators (one pass, O(D)
memory). Quantiles q01/q99 are estimated from a fixed-size per-dimension reservoir
sample, which keeps memory bounded regardless of dataset size while staying within
a fraction of a percent of the true quantile for normalization purposes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Per-dimension reservoir size for quantile estimation. 50k samples gives a
# tight q01/q99 estimate while bounding memory to ~50k * D floats.
DEFAULT_RESERVOIR = 50_000


class StreamingStats:
    """Accumulate stats for one 2D feature (rows x dims) across many chunks."""

    def __init__(self, dim: int, reservoir_size: int = DEFAULT_RESERVOIR, seed: int = 0):
        self.dim = dim
        self.count = 0
        self._sum = np.zeros(dim, dtype=np.float64)
        self._sumsq = np.zeros(dim, dtype=np.float64)
        self._min = np.full(dim, np.inf, dtype=np.float64)
        self._max = np.full(dim, -np.inf, dtype=np.float64)
        self.reservoir_size = reservoir_size
        self._reservoir = np.empty((reservoir_size, dim), dtype=np.float64)
        self._filled = 0
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def update(self, data: np.ndarray) -> None:
        """data: (n, dim) array of one chunk's rows."""
        if data.size == 0:
            return
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, self.dim)
        n = data.shape[0]
        self.count += n
        self._sum += data.sum(axis=0)
        self._sumsq += (data * data).sum(axis=0)
        self._min = np.minimum(self._min, data.min(axis=0))
        self._max = np.maximum(self._max, data.max(axis=0))
        self._update_reservoir(data)

    def _update_reservoir(self, data: np.ndarray) -> None:
        n = data.shape[0]
        # Fill phase.
        if self._filled < self.reservoir_size:
            take = min(n, self.reservoir_size - self._filled)
            self._reservoir[self._filled : self._filled + take] = data[:take]
            self._filled += take
            self._seen += take
            data = data[take:]
            if data.shape[0] == 0:
                return
        # Replacement phase (Algorithm R).
        for row in data:
            self._seen += 1
            j = self._rng.integers(0, self._seen)
            if j < self.reservoir_size:
                self._reservoir[j] = row

    def finalize(self, q_low: float = 0.01, q_high: float = 0.99) -> dict[str, list[float]]:
        if self.count == 0:
            zeros = [0.0] * self.dim
            return {k: list(zeros) for k in ("mean", "std", "min", "max", "q01", "q99")}
        mean = self._sum / self.count
        var = np.maximum(self._sumsq / self.count - mean * mean, 0.0)
        std = np.sqrt(var)
        sample = self._reservoir[: self._filled]
        q01 = np.quantile(sample, q_low, axis=0)
        q99 = np.quantile(sample, q_high, axis=0)
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


class StatsCollector:
    """Collects stats for state, action, timestamp, and (optional) relative arm.

    Mirrors the structure of meta/stats.json produced by the Galaxea converter:
        { "observation.state": {...}, "action": {...}, "timestamp": {...} }
    and meta/relative_stats.json keyed by arm modality keys.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        relative_arm_slices: Optional[dict[str, tuple[int, int]]] = None,
        q_low: float = 0.01,
        q_high: float = 0.99,
    ):
        self.q_low = q_low
        self.q_high = q_high
        self.state = StreamingStats(state_dim)
        self.action = StreamingStats(action_dim)
        self.timestamp = StreamingStats(1)
        # relative action = action[t] - state[t] for arm dims, per arm key.
        self.relative_arm_slices = relative_arm_slices or {}
        self.relative = {
            key: StreamingStats(end - start)
            for key, (start, end) in self.relative_arm_slices.items()
        }

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        timestamp: np.ndarray,
    ) -> None:
        self.state.update(state)
        self.action.update(action)
        self.timestamp.update(np.asarray(timestamp, dtype=np.float64).reshape(-1, 1))
        for key, (start, end) in self.relative_arm_slices.items():
            rel = action[:, start:end] - state[:, start:end]
            self.relative[key].update(rel)

    def finalize_stats(self) -> dict:
        return {
            "observation.state": self.state.finalize(self.q_low, self.q_high),
            "action": self.action.finalize(self.q_low, self.q_high),
            "timestamp": self.timestamp.finalize(self.q_low, self.q_high),
        }

    def finalize_relative_stats(self) -> dict:
        return {key: s.finalize(self.q_low, self.q_high) for key, s in self.relative.items()}


def compute_episode_stats(
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    """Per-episode min/max/mean/std/count for state/action/timestamp.

    This matches the upstream HF `lerobot` episodes_stats.jsonl schema (used by
    openpi/pi05): each numeric feature carries min/max/mean/std plus a 'count'
    (number of frames). Image/video features are intentionally omitted — GR00T's
    own stats.json omits them too, openpi normalizes images internally, and the
    upstream aggregate_stats takes the union of keys, so missing image keys are
    fine and we avoid decoding any video.

    Values are kept as 1-D lists (length == feature dim) so json round-trips
    cleanly; 'count' is a 1-element list per the upstream shape requirement.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for key, arr in (
        ("observation.state", state),
        ("action", action),
        ("timestamp", np.asarray(timestamp, dtype=np.float64).reshape(-1, 1)),
    ):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[0]
        out[key] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(n)],
        }
    return out
