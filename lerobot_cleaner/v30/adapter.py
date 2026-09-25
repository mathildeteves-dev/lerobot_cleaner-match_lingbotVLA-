"""Backward-compatible frame bridge to UnifiedEpisode and TrajectoryView."""
from lerobot_cleaner.adapters.episode import UnifiedEpisode


class V30Adapter:
    @staticmethod
    def to_trajectory(frame, fps, state_column="observation.state", action_column="action"):
        if "episode_index" in frame and frame["episode_index"].nunique() > 1:
            raise ValueError("V30Adapter requires a single episode slice")
        indices = frame["frame_index"].tolist() if "frame_index" in frame else list(range(len(frame)))
        return UnifiedEpisode(ref=None, df=frame, keep_indices=indices).to_trajectory(
            fps, state_column, action_column)
