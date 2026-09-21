"""Static trimming consumes a plan; it never repeats motion detection."""
from ..trajectory.frame_filter import apply_frame_selection


def apply_static_trim(episode, plan):
    # Trim and arbitrary drops are composed into exactly one shared frame mask.
    return apply_frame_selection(episode, plan)
