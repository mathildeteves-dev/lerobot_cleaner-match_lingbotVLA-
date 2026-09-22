"""Legacy name for the pure visual integrity checker."""
from .visual import check_visual


def check_video(observations, decoded=False):
    finding = check_visual(observations, decoded)
    finding.rule = "video_integrity"
    return finding
