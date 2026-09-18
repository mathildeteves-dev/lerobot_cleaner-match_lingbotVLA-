"""Dataset-level ordering and statistics from finalized data."""
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from test_numeric_stages import dataset

from lerobot_cleaner.v21.config import CleaningConfig
from lerobot_cleaner.v21.rules import build_finalizers
from lerobot_cleaner.v21.rules.base import CheckRule, TransformRule
from lerobot_cleaner.v21.rules.finalizers.base import DatasetFinalizer
from lerobot_cleaner.v21.rules.finalizers.lingbot_norm_stats import LingBotNormStatsFinalizer
from lerobot_cleaner.v21.rules.finalizers.reindex_and_restats import ReindexAndRestatsRule
from lerobot_cleaner.v21.writer import DatasetWriter


def test_finalizers_are_separate_dataset_operations_in_dependency_order():
    calls = []
    writer = Mock(out=Path("clean"))
    writer.episodes_out = [{"episode_index": 0, "length": 2}]
    writer.finalize_metadata.side_effect = lambda: calls.append("metadata")
    writer.rebuild_stats.side_effect = lambda: calls.append("rebuild_stats")
    writer.finalize_stats.side_effect = lambda: calls.append("stats")
    writer.verify_episode_alignment.side_effect = lambda *args: calls.append("verify")
    report = Mock(alignment_errors=[])
    finalizers = build_finalizers(CleaningConfig(), lambda path: calls.append(("lingbot", path)))
    assert [f.name for f in finalizers] == ["reindex", "lerobot_metadata", "stats", "lingbot_norm_stats"]
    assert all(isinstance(f, DatasetFinalizer) and not isinstance(f, (CheckRule, TransformRule)) for f in finalizers)
    for finalizer in finalizers:
        finalizer.finalize([], writer, report)
    assert calls == ["metadata", "verify", "rebuild_stats", "stats", ("lingbot", Path("clean"))]


def test_legacy_r8_facade_runs_the_entire_chain():
    writer = Mock()
    writer.episodes_out = []
    ReindexAndRestatsRule(CleaningConfig().rules.reindex_and_restats).apply([], writer, Mock())
    writer.finalize_metadata.assert_called_once_with()
    writer.rebuild_stats.assert_called_once_with()
    writer.finalize_stats.assert_called_once_with()


@pytest.mark.parametrize("errors,episodes", [(["bad alignment"], [1]), ([], [])])
def test_lingbot_adapter_rejects_incomplete_input(errors, episodes):
    adapter = Mock()
    with pytest.raises(ValueError):
        LingBotNormStatsFinalizer(adapter).finalize([], Mock(episodes_out=episodes), Mock(alignment_errors=errors))
    adapter.assert_not_called()


def test_rebuild_stats_reads_final_output_and_does_not_accumulate_twice():
    writer = DatasetWriter.__new__(DatasetWriter)
    writer.source = dataset()
    writer.out = Path("clean")
    writer.data_pattern = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    writer.chunk_size = 1000
    writer.episodes_out = [{"episode_index": 0, "length": 2}]
    df = pd.DataFrame({"observation.state": [np.array([1., 2.]), np.array([3., 4.])],
                       "action": [np.array([5., 6.]), np.array([7., 8.])], "timestamp": [0., .1]})
    with patch("lerobot_cleaner.v21.writer.pd.read_parquet", return_value=df) as read:
        writer.rebuild_stats()
        writer.rebuild_stats()
    read.assert_called_with(Path("clean/data/chunk-000/episode_000000.parquet"))
    assert len(writer.episodes_stats_out) == 1
    stats = writer.stats.finalize_stats()
    np.testing.assert_allclose(stats["observation.state"]["mean"], [2, 3])
    np.testing.assert_allclose(stats["action"]["mean"], [6, 7])


def test_stats_reject_metadata_length_mismatch():
    writer = DatasetWriter.__new__(DatasetWriter)
    writer.source = dataset()
    writer.out = Path("clean")
    writer.data_pattern = "{episode_index}.parquet"
    writer.chunk_size = 1000
    writer.episodes_out = [{"episode_index": 0, "length": 3}]
    with patch("lerobot_cleaner.v21.writer.pd.read_parquet", return_value=pd.DataFrame({"timestamp": [0]})):
        with pytest.raises(ValueError, match="length differs"):
            writer.rebuild_stats()


@pytest.mark.parametrize("bad", ["missing", "nonfinite"])
def test_stats_reject_invalid_final_numeric_data(bad):
    writer = DatasetWriter.__new__(DatasetWriter)
    writer.source = dataset()
    writer.out = Path("clean")
    writer.data_pattern = "{episode_index}.parquet"
    writer.chunk_size = 1000
    writer.episodes_out = [{"episode_index": 0, "length": 1}]
    df = pd.DataFrame({"observation.state": [np.array([np.nan, 1])],
                       "action": [np.array([0, 1])], "timestamp": [0.]})
    if bad == "missing":
        df = df.drop(columns=["action"])
    with patch("lerobot_cleaner.v21.writer.pd.read_parquet", return_value=df):
        with pytest.raises(ValueError, match="missing required|non-finite"):
            writer.rebuild_stats()
