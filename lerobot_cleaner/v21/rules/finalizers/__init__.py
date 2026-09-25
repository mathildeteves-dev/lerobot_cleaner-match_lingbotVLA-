"""Ordered dataset finalization; never included in per-episode rule lists."""
from .base import DatasetFinalizer
from .lerobot_metadata import LeRobotMetadataFinalizer
from .lingbot_norm_stats import LingBotNormStatsFinalizer
from .reindex import ReindexFinalizer
from .stats import StatsFinalizer


def dataset_finalizers(config, lingbot_normalizer=None) -> list[DatasetFinalizer]:
    finalizers = [ReindexFinalizer(), LeRobotMetadataFinalizer(config), StatsFinalizer()]
    if lingbot_normalizer is not None:
        finalizers.append(LingBotNormStatsFinalizer(lingbot_normalizer))
    return finalizers
