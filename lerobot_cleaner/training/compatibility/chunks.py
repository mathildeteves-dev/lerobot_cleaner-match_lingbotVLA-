"""Exact forward integer-delta query semantics, LeRobot 0.4.2."""
import numpy as np


def sample_indices(length, timestep, horizon, offset=0):
    if length < 1 or horizon < 1 or not 0 <= timestep < length:
        raise ValueError("Invalid episode/timestep/horizon")
    requested = timestep + np.arange(horizon, dtype=np.int64)
    return np.clip(requested, 0, length - 1) + offset, (requested < 0) | (requested >= length)


def chunk_statistics(length, horizon, include_timesteps=False):
    if length < 1 or horizon < 1:
        raise ValueError("Episode length and horizon must be positive")
    valid = np.minimum(horizon, length - np.arange(length, dtype=np.int64))
    padded = horizon - valid
    slots = int(length * horizon)
    total_pad = int(padded.sum())
    result = {"episode_length": length, "num_training_samples": length, "action_chunk_size": horizon,
        "total_action_slots": slots, "usable_action_steps": int(valid.sum()), "padded_action_steps": total_pad,
        "action_chunk_padding_ratio": total_pad / slots, "boundary_padding_steps": total_pad,
        "boundary_padding_ratio": total_pad / slots, "non_boundary_padding_steps": 0,
        "fully_unpadded_samples": int((padded == 0).sum()), "partially_padded_samples": int(((padded > 0) & (padded < horizon)).sum()),
        "fully_padded_samples": int((padded == horizon).sum()), "sample_padding_ratio": float(np.mean(padded > 0)),
        "definition": "padded temporal slots / requested temporal slots, not dimension padding; every row is one sample"}
    if include_timesteps:
        result["padding_by_timestep"] = (padded / horizon).tolist()
    return result


def summarize_chunks(episodes):
    rows = [e["action_chunk"] for e in episodes if "action_chunk" in e]
    total_slots = sum(r["total_action_slots"] for r in rows)
    padded = sum(r["padded_action_steps"] for r in rows)
    ratios = [r["action_chunk_padding_ratio"] for r in rows]
    return {"total_samples": sum(r["num_training_samples"] for r in rows), "total_action_slots": total_slots,
        "total_padded_action_slots": padded, "usable_action_steps": total_slots-padded,
        "action_chunk_padding_ratio": padded/total_slots if total_slots else None,
        "boundary_padding_ratio": padded/total_slots if total_slots else None,
        "mean_padding_ratio": float(np.mean(ratios)) if ratios else None,
        "median_padding_ratio": float(np.median(ratios)) if ratios else None,
        "p95_padding_ratio": float(np.percentile(ratios, 95)) if ratios else None,
        "distribution_unit": "episode ratios, equal episode weight; overall ratio is slot weighted",
        **{f"episodes_with_padding_ratio_gt_{str(t).replace('.', '_')}": sum(r > t for r in ratios) for t in (.25, .5, .75)}}
