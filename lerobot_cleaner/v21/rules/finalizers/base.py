"""Dataset-level contract, separate from episode checks and transforms."""
from abc import ABC, abstractmethod


class DatasetFinalizer(ABC):
    name: str

    @abstractmethod
    def finalize(self, results, writer, report) -> None:
        """Run once after every episode has finished processing."""

    def apply(self, results, writer, report) -> None:
        """Compatibility spelling for dataset-level callers."""
        self.finalize(results, writer, report)
