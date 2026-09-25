"""LingBot-VLA integration: v2.1→v3.0 export, mapping validation, config generation.

Modules:
- ``vla``:     export a cleaned LeRobot v2.1 dataset as LeRobot v3.0 for LingBot-VLA.
- ``config``:  static validation of robot/train mapping YAMLs against a real dataset.
- ``generate``: generate a robot + train config pair from a per-embodiment spec.
"""

from lerobot_cleaner.v30.lingbot.config import validate_mapping
from lerobot_cleaner.v30.lingbot.generate import generate_configs
from lerobot_cleaner.v30.lingbot.vla import export_lingbot_dataset

__all__ = ["export_lingbot_dataset", "generate_configs", "validate_mapping"]
