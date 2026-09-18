"""Optional LingBot normalization integration, after dataset finalization.

Inject a callable accepting the finalized dataset Path. It must invoke the
LingBot-specific normalization workflow with validated robot/train mappings and
validate its output. LeRobot stats must not be reused as LingBot norm stats.
The repository's scripts/run_lingbot_norm.py provides the existing v3 workflow;
a v2.1-compatible adapter is required before enabling this hook for this engine.
"""
from .base import DatasetFinalizer


class LingBotNormStatsFinalizer(DatasetFinalizer):
    name = "lingbot_norm_stats"

    def __init__(self, normalizer):
        if not callable(normalizer):
            raise TypeError("LingBot finalization requires a configured normalization adapter")
        self.normalizer = normalizer

    def finalize(self, results, writer, report):
        if report.alignment_errors:
            raise ValueError("Cannot compute LingBot normalization with alignment errors")
        if not writer.episodes_out:
            raise ValueError("Cannot compute LingBot normalization for an empty dataset")
        self.normalizer(writer.out)
