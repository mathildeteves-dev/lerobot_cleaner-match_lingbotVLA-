"""Select semantic adapters after the official storage layer has loaded data."""
from .groot import GrootAdapter
from .lerobot_v3 import LeRobotAdapter
from .lingbot import LingBotAdapter


def v3_adapter(root, config, *, info=None, writer=None, storage=None):
    kind = config.semantic_adapter
    if kind == "auto":
        kind = "lingbot" if config.robot_config is not None else ("groot" if config.modality_config is not None else "lerobot")
    if kind == "lingbot":
        return LingBotAdapter(root, config.robot_config, config, writer=writer, storage=storage)
    if kind == "groot":
        return GrootAdapter(root, config, writer=writer, storage=storage)
    return LeRobotAdapter(root, config, writer=writer, storage=storage)
