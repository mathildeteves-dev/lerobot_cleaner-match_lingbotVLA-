"""LeRobot feature semantics over the official storage abstraction."""
from lerobot_cleaner.storage import OfficialStorage

from .base import DatasetAdapter
from .schema import FeatureSchema, FeatureSlice


class LeRobotAdapter(DatasetAdapter):
    """Feature semantics over an official storage handle, independent of shards."""
    def __init__(self, root, config=None, *, info=None, writer=None, storage=None):
        if config is None:
            from lerobot_cleaner.v30.v3 import V3Config
            config = V3Config(engine="streaming")
        self.config = config
        self._owns_storage = storage is None
        self.storage = storage if storage is not None else OfficialStorage(root, config)
        self.root = self.storage.root
        # Official metadata is authoritative; info is only a compatibility argument.
        self.info = self.storage.info
        self.fps = self.info["fps"]
        self._writer = writer

    def close(self):
        if self._owns_storage:
            self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _feature(self, modality, column):
        spec = self.info["features"].get(column)
        if spec is None:
            return ()
        if len(spec["shape"]) != 1:
            raise ValueError(f"Expected vector feature: {column}")
        return (FeatureSchema(column, modality, (FeatureSlice(column, 0, spec["shape"][0]),)),)

    def get_state_features(self):
        return self._feature("state", self.config.quality.state_column)

    def get_action_features(self):
        return self._feature("action", self.config.quality.action_column)

    def get_camera_features(self):
        return tuple(FeatureSchema(name, "video", camera_column=name)
                     for name, spec in self.info["features"].items() if spec["dtype"] == "video")

    def describe(self):
        return {**super().describe(), "reader_backend": "lerobot",
                "storage": "LeRobotDataset", "storage_version": "v3.0",
                "reader_memory": "official HF initialization plus episode slices; not bounded by batch_rows"}

    def iter_episodes(self):
        for index in range(self.info["total_episodes"]):
            yield self.read_episode(index)

    def read_episode(self, episode_id):
        table, row = self.storage.episode(episode_id)
        return self.from_frame(table.to_pandas(), metadata=row)

    def from_frame(self, frame, *, ref=None, metadata=None):
        episode = super().from_frame(frame, ref=ref, metadata=metadata)
        # v3 is row-preserving; retain identity columns to guard staged writes.
        keys = [key for key in ("index", "episode_index", "frame_index", "timestamp", "task_index") if key in frame]
        episode.metadata["row_identity"] = frame[keys].copy(deep=True)
        return episode

    def write_episode(self, episode, *, writer=None):
        identity = episode.metadata.get("row_identity")
        if episode.dropped or episode.keep_indices != list(range(episode.ref.length)):
            raise ValueError("v3 adapter does not allow dropping or reordering frames")
        if identity is None or not episode.df[list(identity.columns)].equals(identity):
            raise ValueError("v3 adapter requires unchanged row identity/timestamps")
        return super().write_episode(episode, writer=writer)


# Backward-compatible import name for the former v3 adapter.
LeRobotV3Adapter = LeRobotAdapter
