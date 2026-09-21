"""GR00T semantic metadata declarations; no dataset loading."""
from dataclasses import dataclass
from typing import Any

META_DIR = "meta"
INFO_FILE = "info.json"
MODALITY_FILE = "modality.json"
EPISODES_FILE = "episodes.jsonl"
TASKS_FILE = "tasks.jsonl"
STATS_FILE = "stats.json"
RELATIVE_STATS_FILE = "relative_stats.json"

# Standard parquet columns that carry the state/action vectors.
STATE_COL = "observation.state"
ACTION_COL = "action"


@dataclass
class ModalitySlice:
    """A named contiguous slice within the state/action vector."""

    modality: str  # "state" | "action"
    key: str  # e.g. "left_gripper"
    start: int
    end: int

    @property
    def column(self) -> str:
        return STATE_COL if self.modality == "state" else ACTION_COL


class ModalityResolver:
    """Resolve dotted modality keys (``state.left_gripper``) to vector slices."""

    def __init__(self, modality_meta: dict[str, Any]):
        self.meta = modality_meta

    def resolve(self, dotted_key: str) -> ModalitySlice:
        if "." not in dotted_key:
            raise ValueError(
                f"Modality target '{dotted_key}' must be of the form "
                f"'<state|action>.<key>', e.g. 'state.left_gripper'."
            )
        modality, key = dotted_key.split(".", 1)
        if modality not in ("state", "action"):
            raise ValueError(f"Unknown modality '{modality}' in '{dotted_key}'.")
        if modality not in self.meta or key not in self.meta[modality]:
            available = list(self.meta.get(modality, {}).keys())
            raise KeyError(
                f"Key '{key}' not in modality.json[{modality}]. Available: {available}"
            )
        block = self.meta[modality][key]
        return ModalitySlice(modality, key, int(block["start"]), int(block["end"]))

    def video_keys(self) -> list[str]:
        return list(self.meta.get("video", {}).keys())

    def video_original_key(self, key: str) -> str:
        return self.meta["video"][key].get("original_key", key)

    def state_dim(self) -> int:
        return max((b["end"] for b in self.meta.get("state", {}).values()), default=0)

    def action_dim(self) -> int:
        return max((b["end"] for b in self.meta.get("action", {}).values()), default=0)

    def arm_keys(self, modality: str = "state") -> list[str]:
        """Keys whose name contains 'arm' — used for relative_stats."""
        return [k for k in self.meta.get(modality, {}) if "arm" in k.lower()]


