"""Legacy R8 compatibility facade for the dataset-level finalizer chain."""
from . import dataset_finalizers
from .base import DatasetFinalizer


def reindex_and_restats(results, writer, report, config):
    for finalizer in dataset_finalizers(config):
        finalizer.finalize(results, writer, report)


class ReindexAndRestatsRule(DatasetFinalizer):
    name = "reindex_and_restats"

    def __init__(self, config):
        self.config = config

    def finalize(self, results, writer, report):
        reindex_and_restats(results, writer, report, self.config)
