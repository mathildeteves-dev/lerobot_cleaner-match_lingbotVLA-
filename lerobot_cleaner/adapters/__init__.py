"""Source adapters → canonical schema → episode assembly → quality arrays."""
from lerobot_cleaner.core.physical import PhysicalSemantics, Unit, Representation, ControlType, CoordinateFrame
from .base import DatasetAdapter, Feature, FeatureSlice
from .schema import FeatureSchema, CanonicalFeatureSchema, VisualFeatureSchema, VisualFeature, VisualSource
from lerobot_cleaner.core.language import LanguageFeature, TaskInfo, EpisodeTask
from .resolver import FeatureResolver
from .builder import AssemblyPolicy, EpisodeBuilder, EpisodeAssembler, EpisodeAdapter
from .episode import UnifiedEpisode
from .groot import GrootAdapter
from .generic import GenericMappingAdapter
from .mapping import FeatureSpec, SourceSpec


def __getattr__(name):
    if name == "LingBotAdapter":
        from .lingbot import LingBotAdapter
        return LingBotAdapter
    raise AttributeError(name)

from .lerobot_v3 import LeRobotAdapter, LeRobotV3Adapter

__all__ = ["PhysicalSemantics", "Unit", "Representation", "ControlType", "CoordinateFrame", "VisualFeatureSchema", "VisualFeature", "VisualSource", "GenericMappingAdapter", "FeatureSpec", "SourceSpec", "LanguageFeature", "TaskInfo", "EpisodeTask", "DatasetAdapter", "Feature", "FeatureSchema", "FeatureSlice",
           "CanonicalFeatureSchema", "FeatureResolver", "AssemblyPolicy",
           "EpisodeBuilder", "EpisodeAssembler", "EpisodeAdapter", "UnifiedEpisode",
           "GrootAdapter", "LingBotAdapter", "LeRobotAdapter", "LeRobotV3Adapter"]
