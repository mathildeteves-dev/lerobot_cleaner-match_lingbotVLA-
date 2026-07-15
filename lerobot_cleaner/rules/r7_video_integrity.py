"""R7: video integrity — every view exists, decodes, and has consistent frame count."""

from __future__ import annotations

from lerobot_cleaner.rules.base import Rule
from lerobot_cleaner.types import EpisodeWork
from lerobot_cleaner.video_utils import VideoError, count_frames, is_decodable


class VideoIntegrityRule(Rule):
    name = "video_integrity"

    def apply(self, work: EpisodeWork) -> None:
        counts: dict[str, int] = {}
        for vkey, vpath in work.ref.video_paths.items():
            if not vpath.exists():
                work.drop(f"missing video {vkey}: {vpath}")
                self.stats["episodes_dropped_missing"] += 1
                return
            if self.config.check_decodable and not is_decodable(vpath):
                work.drop(f"undecodable video {vkey}")
                self.stats["episodes_dropped_undecodable"] += 1
                return
            try:
                counts[vkey] = count_frames(vpath)
            except VideoError:
                work.drop(f"cannot count frames for {vkey}")
                self.stats["episodes_dropped_undecodable"] += 1
                return

        if self.config.check_frame_count_consistency and len(set(counts.values())) > 1:
            work.drop(f"inconsistent frame counts across views: {counts}")
            self.stats["episodes_dropped_inconsistent"] += 1
