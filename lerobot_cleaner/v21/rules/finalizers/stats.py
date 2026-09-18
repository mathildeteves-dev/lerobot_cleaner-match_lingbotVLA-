"""Recompute LeRobot statistics from the final, reindexed parquet data."""
from .base import DatasetFinalizer


class StatsFinalizer(DatasetFinalizer):
    name = "stats"

    def finalize(self, results, writer, report):
        writer.rebuild_stats()
        writer.finalize_stats()
