"""Publish kept episodes with contiguous dataset and frame indices."""
from .base import DatasetFinalizer


class ReindexFinalizer(DatasetFinalizer):
    name = "reindex"

    def finalize(self, results, writer, report):
        results.sort(key=lambda result: result.src_index)
        new_index = 0
        for result in results:
            report.record_episode(result)
            if result.dropped:
                continue
            writer.finalize_staged_episode(
                new_index, result.staged_parquet, result.staged_videos, result.tasks,
                collect_stats=False,
            )
            new_index += 1
