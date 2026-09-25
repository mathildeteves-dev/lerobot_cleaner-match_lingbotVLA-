"""Dataset semantics and episode I/O contract, independent of modality.json."""
from abc import ABC, abstractmethod
from dataclasses import asdict

from .schema import Feature, FeatureSlice
from .resolver import FeatureResolver
from .builder import EpisodeBuilder

from lerobot_cleaner.adapters.episode import UnifiedEpisode


class DatasetAdapter(ABC):
    """Writes require a pipeline-owned staging writer; never overwrite raw data.

    A writer is a callable accepting UnifiedEpisode. Dataset metadata/statistics
    remain the responsibility of the storage engine's dataset-level finalizer.
    """
    @abstractmethod
    def get_state_features(self): ...

    @abstractmethod
    def get_action_features(self): ...

    @abstractmethod
    def get_camera_features(self): ...

    @abstractmethod
    def read_episode(self, episode_id): ...

    def get_visual_features(self):
        """Storage-independent camera observations (legacy camera API remains supported)."""
        return self.get_feature_schema().visual

    def get_feature_schema(self):
        return FeatureResolver().resolve(self)

    def get_language_feature(self):
        from lerobot_cleaner.core.language import LanguageFeature
        return LanguageFeature()

    def get_task_catalog(self):
        from .language import task_catalog
        if not hasattr(self, "_task_catalog"):
            storage = getattr(self, "storage", None)
            table = getattr(storage, "tasks", getattr(self, "tasks_meta", None))
            self._task_catalog = task_catalog(table)
        return self._task_catalog

    def episode_builder(self, policy=None):
        return EpisodeBuilder(self.get_feature_schema(), self.fps, policy)

    def from_frame(self, frame, *, ref=None, metadata=None):
        info = getattr(self, "info", {})
        meta = {**(metadata or {}), "source_root": str(self.root)}
        meta["task_table_expected_count"] = info.get("total_tasks")
        meta["dataset_language"] = {key: info[key] for key in
            ("task", "language", "instruction", "prompt") if key in info}
        return self.episode_builder().build(
            frame, ref=ref, metadata=meta, task_catalog=self.get_task_catalog())

    def describe(self):
        return {"type": type(self).__name__,
                "state_features": [asdict(feature) for feature in self.get_state_features()],
                "action_features": [asdict(feature) for feature in self.get_action_features()],
                "camera_features": [asdict(feature) for feature in self.get_camera_features()],
                "language": asdict(self.get_feature_schema().language),
                "action_values": "stored values; subtract_state is metadata, not applied by cleaning"}

    def write_episode(self, episode, *, writer=None):
        if not isinstance(episode, UnifiedEpisode):
            raise TypeError("write_episode requires UnifiedEpisode")
        if episode.metadata.get("source_root", str(self.root)) != str(self.root):
            raise ValueError("Episode belongs to a different dataset")
        if len(episode.df) != len(episode.keep_indices):
            raise ValueError("Episode frame/keep_indices mismatch")
        sink = writer if writer is not None else getattr(self, "_writer", None)
        if sink is None:
            raise ValueError("No output writer bound; adapters never write to the input dataset")
        return sink(episode)
