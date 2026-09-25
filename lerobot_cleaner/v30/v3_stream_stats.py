"""Bounded-memory numeric summaries and stable merged moments.

The reservoir is exact Algorithm R with vectorized draws; when a batch selects
the same reservoir slot repeatedly, only its last replacement is retained.
No Python loop over individual frames and no full-dataset numeric array.
"""

import numpy as np


class NumericSummary:
    def __init__(self, dim):
        self.dim = dim
        self.nonfinite = 0
        self.low = np.full(dim, np.inf)
        self.high = np.full(dim, -np.inf)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        self.nonfinite += int((~finite).sum())
        self.low = np.minimum(self.low, np.where(finite, values, np.inf).min(axis=0))
        self.high = np.maximum(self.high, np.where(finite, values, -np.inf).max(axis=0))

    def result(self):
        result = {"dimensions": self.dim, "nonfinite": self.nonfinite}
        if not self.nonfinite:
            result.update(min=self.low.tolist(), max=self.high.tolist())
        return result


class OnlineStats:
    def __init__(self, dim, capacity=32768, seed=0):
        self.count = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.low = np.full(dim, np.inf)
        self.high = np.full(dim, -np.inf)
        self.sample = np.empty((capacity, dim))
        self.rng = np.random.default_rng(seed)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise ValueError("Cannot accumulate non-finite output statistics")
        n, old = len(values), self.count
        mean = values.mean(axis=0)
        delta = mean - self.mean
        self.mean += delta * (n / (old + n))
        self.m2 += ((values - mean) ** 2).sum(axis=0) + delta**2 * (old * n / (old + n))
        self.low = np.minimum(self.low, values.min(axis=0))
        self.high = np.maximum(self.high, values.max(axis=0))
        capacity = len(self.sample)
        fill = min(n, max(0, capacity - old))
        if fill:
            self.sample[old : old + fill] = values[:fill]
        if fill < n:
            slots = self.rng.integers(0, np.arange(old + fill + 1, old + n + 1))
            selected = np.flatnonzero(slots < capacity)
            if len(selected):
                unique_slots, reverse_indices = np.unique(slots[selected][::-1], return_index=True)
                self.sample[unique_slots] = values[fill + selected[::-1][reverse_indices]]
        self.count += n

    def result(self):
        if not self.count:
            raise ValueError("No rows for statistics")
        result = {
            "count": [self.count],
            "mean": self.mean.tolist(),
            "std": np.sqrt(np.maximum(0, self.m2 / self.count)).tolist(),
            "min": self.low.tolist(),
            "max": self.high.tolist(),
        }
        qs = [1, 10, 50, 90, 99]
        values = np.quantile(
            self.sample[: min(self.count, len(self.sample))], np.asarray(qs) / 100, axis=0
        )
        result.update({f"q{q:02d}": value.tolist() for q, value in zip(qs, values)})
        return result
