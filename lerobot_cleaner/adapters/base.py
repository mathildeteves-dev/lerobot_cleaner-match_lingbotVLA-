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

    def get_feature_schema(self):
        return FeatureResolver().resolve(self)

    def episode_builder(self, policy=None):
        return EpisodeBuilder(self.get_feature_schema(), self.fps, policy)

    def from_frame(self, frame, *, ref=None, metadata=None):
        return self.episode_builder().build(
            frame, ref=ref, metadata={**(metadata or {}), "source_root": str(self.root)})

    def describe(self):
        return {"type": type(self).__name__,
                "state_features": [asdict(feature) for feature in self.get_state_features()],
                "action_features": [asdict(feature) for feature in self.get_action_features()],
                "camera_features": [asdict(feature) for feature in self.get_camera_features()],
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
