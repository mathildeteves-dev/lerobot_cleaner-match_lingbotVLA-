"""Rule registry and ordered construction.

Per-episode rules run in this fixed order. R8 (reindex/restats) is NOT a
per-episode rule — it is performed by the writer/pipeline after all episodes are
processed — so it does not appear here.
"""

from __future__ import annotations

from lerobot_cleaner.rules.base import Rule
from lerobot_cleaner.rules.r1_timestamp import TimestampAlignmentRule
from lerobot_cleaner.rules.r2_static_trim import StaticFrameTrimRule
from lerobot_cleaner.rules.r3_gripper import GripperBinarizeRule
from lerobot_cleaner.rules.r4_video_roi import VideoRoiCropRule
from lerobot_cleaner.rules.r5_length import EpisodeLengthFilterRule
from lerobot_cleaner.rules.r6_numeric import NumericSanityRule
from lerobot_cleaner.rules.r7_video_integrity import VideoIntegrityRule

# Order matters:
#   integrity/alignment first (drop bad episodes cheaply),
#   then value transforms, then length filter last (length depends on trims).
RULE_ORDER = [
    ("video_integrity", VideoIntegrityRule),
    ("timestamp_alignment", TimestampAlignmentRule),
    ("numeric_sanity", NumericSanityRule),
    ("gripper_binarize", GripperBinarizeRule),
    ("static_frame_trim", StaticFrameTrimRule),
    ("video_roi_crop", VideoRoiCropRule),
    ("episode_length_filter", EpisodeLengthFilterRule),
]

__all__ = ["Rule", "RULE_ORDER", "build_rules"]


def build_rules(config, dataset, outlier_bounds=None) -> list[Rule]:
    """Instantiate enabled rules in canonical order."""
    rules: list[Rule] = []
    for name, cls in RULE_ORDER:
        rule_cfg = getattr(config.rules, name)
        if not rule_cfg.enabled:
            continue
        if name == "numeric_sanity":
            rules.append(cls(rule_cfg, dataset, outlier_bounds=outlier_bounds))
        else:
            rules.append(cls(rule_cfg, dataset))
    return rules
