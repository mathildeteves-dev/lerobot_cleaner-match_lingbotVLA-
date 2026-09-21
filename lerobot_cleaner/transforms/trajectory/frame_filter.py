"""One source-frame selection for every synchronized modality."""
import numpy as np


def apply_frame_selection(episode, plan):
    count = len(episode.df)
    keep = np.ones(count, dtype=bool)
    if plan.trim is not None:
        start, stop = plan.trim
        if not 0 <= start < stop <= count:
            raise ValueError("Invalid planned trim interval")
        keep[:start] = False
        keep[stop:] = False
    if any(type(index) is not int or not 0 <= index < count for index in plan.drop_frames):
        raise ValueError("Invalid planned source-frame index")
    keep[plan.drop_frames] = False
    # A row contains every non-video modality. Video/depth stored externally use
    # these SAME original source positions at the dataset writer boundary.
    episode.restrict_to(keep)
    episode.metadata["source_frame_selection"] = list(episode.keep_indices)
    episode.metadata["source_timestamps"] = episode.df.timestamp.tolist()
    return int(count - len(episode.df))
