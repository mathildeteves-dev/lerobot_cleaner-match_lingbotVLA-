"""TransformPlan -> independent clean UnifiedEpisode, with a mutation audit trail."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
from .episode_filter import EpisodeFilter
from .trajectory.frame_filter import apply_frame_selection
from .numeric.repair import apply_numeric
from .motion.static_trim import apply_static_trim


def execute_plan(source, plan, info):
    # Checks and policies always see the untouched source object.
    from lerobot_cleaner.adapters.episode_access import episode_identity
    if (source.episode_index != plan.episode_id or len(source.df) != plan.source_frames
            or episode_identity(source) != plan.source_identity):
        raise ValueError("Plan does not match the source episode")
    clean = deepcopy(source)
    if not EpisodeFilter.keep(clean, plan):
        return clean
    removed = apply_static_trim(clean, plan) if plan.trim is not None else apply_frame_selection(clean, plan)
    clean.metadata["source_episode_length"] = source.ref.length
    clean.ref = replace(clean.ref, length=len(clean.df))
    if not len(clean.df):
        raise ValueError("Empty selected episode must be explicitly rejected")
    edits = []
    for edit in plan.numeric:
        count = apply_numeric(clean, edit, info)
        edits.append({"column": edit.column, "operation": edit.operation, "changed_values": count})
    if plan.reindex:
        clean.df["frame_index"] = np.arange(len(clean.df), dtype=np.int64)
        clean.df["timestamp"] = np.arange(len(clean.df), dtype=float) / clean.fps
    clean.metadata["video_crop"] = deepcopy(plan.crop)
    clean.metadata["model_preprocessing"] = deepcopy(plan.preprocessing)
    clean.metadata["transform_plan"] = plan.to_dict()
    clean.transform_results["plan"] = {"changed": plan.changes_data,
        "metrics": {"frames_removed": removed, "numeric": edits, "retimed": plan.reindex}}
    return clean
