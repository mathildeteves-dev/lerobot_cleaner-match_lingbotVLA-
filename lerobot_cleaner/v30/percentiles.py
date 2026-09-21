"""Fit optional source-wide percentile references once, before planning mutations."""
import numpy as np
from lerobot_cleaner.adapters.episode_access import sensor_values
from .planning import resolve_target
from .v3_stream_stats import OnlineStats


def resolve_percentiles(storage, adapter, config):
    requests = [(index, item) for index, item in enumerate(config.transforms.numeric)
                if item.operation == "percentile_clip" and item.low is None]
    if not requests:
        return config, []
    estimates, references = {}, []
    for index, item in requests:
        parts = resolve_target(adapter, item.target)
        estimates[index] = (parts, OnlineStats(sum(part.end-part.start for part in parts), config.quantile_samples))
    for eid in range(storage.info["total_episodes"]):
        episode = adapter.read_episode(eid)
        for index, (parts, stats) in estimates.items():
            values = np.concatenate([sensor_values(episode.df[p.column], storage.info["features"][p.column]["shape"])[:, p.start:p.end]
                                     for p in parts], axis=1)
            stats.update(values[np.isfinite(values).all(axis=1)])
    data = config.model_dump(mode="json")
    for index, item in requests:
        _, stats = estimates[index]
        if not stats.count:
            raise ValueError(f"No finite source samples for percentile target {item.target}")
        sample = stats.sample[:min(stats.count, len(stats.sample))]
        low, high = np.quantile(sample, item.quantiles, axis=0)
        data["transforms"]["numeric"][index].update(low=low.tolist(), high=high.tolist())
        references.append({"target": item.target, "quantiles": item.quantiles,
            "low": low.tolist(), "high": high.tolist(), "finite_source_rows": stats.count,
            "reservoir_rows": len(sample), "seed": 0,
            "exact": stats.count <= len(stats.sample), "population": "finite complete target vectors in original source"})
    return type(config).model_validate(data), references
