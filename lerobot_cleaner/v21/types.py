"""Compatibility exports; episode data is independent of GR00T and LeRobot versions."""
from lerobot_cleaner.adapters.episode import TransformResult, UnifiedEpisode, VideoTransform
from lerobot_cleaner.core.trajectory import CheckResult

EpisodeWork = UnifiedEpisode
__all__ = ["EpisodeWork", "TransformResult", "VideoTransform", "CheckResult"]
