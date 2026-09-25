"""GR00T feature semantics over officially loaded (possibly converted) v3 data."""
import json
from lerobot_cleaner.core.physical import PHYSICAL_FIELDS
from pathlib import Path

from .lerobot_v3 import LeRobotAdapter
from .resolver import FeatureResolver
from .schema import FeatureSchema, FeatureSlice
from .modality import (
    ModalityResolver, ModalitySlice, ACTION_COL, STATE_COL, META_DIR, INFO_FILE,
    MODALITY_FILE, EPISODES_FILE, TASKS_FILE, STATS_FILE, RELATIVE_STATS_FILE,
)


class GrootAdapter(LeRobotAdapter):
    raw_vectors = True

    def __init__(self, root, config=None, *, info=None, writer=None, storage=None):
        super().__init__(root, config, info=info, writer=writer, storage=storage)
        configured = getattr(config, "modality_config", None)
        candidates = ([Path(configured)] if configured is not None else
                      [self.root / "meta/modality.json",
                       self.root / "cleaning_report/source_modality.v21.json"])
        path = next((path for path in candidates if path.is_file()), None)
        if path is None:
            raise FileNotFoundError("GrootAdapter requires modality_config or a preserved modality.json")
        self.modality = json.loads(path.read_text(encoding="utf-8"))
        self.resolver = ModalityResolver(self.modality)

    def _features(self, modality):
        column = STATE_COL if modality == "state" else ACTION_COL
        features = []
        spec = self.info["features"].get(column)
        for key, block in self.modality.get(modality, {}).items():
            start, end = block["start"], block["end"]
            if (spec is None or type(start) is not int or type(end) is not int
                    or len(spec["shape"]) != 1 or not 0 <= start < end <= spec["shape"][0]):
                raise ValueError(f"Invalid GR00T slice: {modality}.{key}")
            features.append(FeatureSchema(f"{modality}.{key}", modality,
                                         (FeatureSlice(column, start, end),),
                                         **{key: block[key] for key in PHYSICAL_FIELDS if key in block}))
        return tuple(features)

    def get_state_features(self):
        return self._features("state")

    def get_action_features(self):
        return self._features("action")

    def get_camera_features(self):
        features = []
        for key in self.resolver.video_keys():
            column = self.resolver.video_original_key(key)
            if self.info["features"].get(column, {}).get("dtype") not in {"video", "image"}:
                raise ValueError(f"Missing GR00T visual source: {column}")
            features.append(FeatureResolver().resolve_visual(
                FeatureSchema(key, "visual", camera_column=column), self.info["features"]))
        return tuple(features)
