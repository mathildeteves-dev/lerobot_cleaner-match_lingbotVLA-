"""Backward-compatible imports for the GR00T adapter and modality resolver."""
from lerobot_cleaner.adapters.episode import EpisodeRef
from lerobot_cleaner.v21.legacy_reader import (
    ACTION_COL,
    EPISODES_FILE,
    INFO_FILE,
    META_DIR,
    MODALITY_FILE,
    RELATIVE_STATS_FILE,
    STATE_COL,
    STATS_FILE,
    TASKS_FILE,
    LegacyGrootAdapter,
    ModalityResolver,
    ModalitySlice,
)

LeRobotDataset = LegacyGrootAdapter
__all__ = ["LeRobotDataset", "EpisodeRef", "ModalityResolver", "ModalitySlice",
           "ACTION_COL", "STATE_COL", "META_DIR", "INFO_FILE", "MODALITY_FILE",
           "EPISODES_FILE", "TASKS_FILE", "STATS_FILE", "RELATIVE_STATS_FILE"]
