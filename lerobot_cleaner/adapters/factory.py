"""Select semantic adapters after the official storage layer has loaded data."""
from .lerobot_v3 import LeRobotAdapter


def v3_adapter(root, config, *, info=None, writer=None, storage=None):
    kind = config.semantic_adapter
    if kind == "auto":
        kind = ("generic" if config.mapping_config is not None else
                "lingbot" if config.robot_config is not None else
                "groot" if config.modality_config is not None else "lerobot")
    if kind == "generic":
        from .generic import GenericMappingAdapter
        return GenericMappingAdapter(root, config.mapping_config, config, writer=writer, storage=storage)
    if kind == "lingbot":
        from .lingbot import LingBotAdapter
        return LingBotAdapter(root, config.robot_config, config, writer=writer, storage=storage)
    if kind == "groot":
        from .groot import GrootAdapter
        return GrootAdapter(root, config, writer=writer, storage=storage)
    return LeRobotAdapter(root, config, writer=writer, storage=storage)
