import numpy as np

from lerobot_cleaner.v21.stats import StreamingStats, compute_episode_stats


def test_compute_episode_stats_matches_numpy():
    rng = np.random.default_rng(3)
    state = rng.normal(size=(50, 4))
    action = rng.normal(size=(50, 4))
    ts = np.arange(50) / 10.0
    out = compute_episode_stats(state, action, ts)
    assert set(out["observation.state"].keys()) == {"min", "max", "mean", "std", "count"}
    assert out["observation.state"]["count"] == [50]
    assert np.allclose(out["observation.state"]["mean"], state.mean(axis=0))
    assert np.allclose(out["action"]["std"], action.std(axis=0))
    # timestamp is 1-D -> stats are length-1 lists
    assert len(out["timestamp"]["mean"]) == 1
    assert out["timestamp"]["count"] == [50]


def test_streaming_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(2.0, 3.0, size=(5000, 4))
    s = StreamingStats(4, reservoir_size=5000)
    # feed in chunks
    for chunk in np.array_split(data, 7):
        s.update(chunk)
    out = s.finalize()
    assert np.allclose(out["mean"], data.mean(axis=0), atol=1e-6)
    assert np.allclose(out["std"], data.std(axis=0), atol=1e-6)
    assert np.allclose(out["min"], data.min(axis=0))
    assert np.allclose(out["max"], data.max(axis=0))
    # full reservoir => exact quantiles
    assert np.allclose(out["q01"], np.quantile(data, 0.01, axis=0), atol=1e-6)


def test_reservoir_quantile_approx():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, size=(200000, 2))
    s = StreamingStats(2, reservoir_size=50000)
    for chunk in np.array_split(data, 20):
        s.update(chunk)
    out = s.finalize()
    true_q99 = np.quantile(data, 0.99, axis=0)
    assert np.allclose(out["q99"], true_q99, atol=0.1)
