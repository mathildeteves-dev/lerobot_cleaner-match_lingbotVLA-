"""Finalize and reopen v3 output using official metadata, never legacy JSONL."""
import shutil
from lerobot_cleaner.storage import OfficialStorage
import numpy as np


def finalize_v3(writer, source, destination, config, expected_episodes, expected_frames):
    if writer is not None:
        writer.finalize()
    # Robot semantics are sidecars; do not copy stale storage metadata/stats.
    modality = source.root / "meta/modality.json"
    if not modality.is_file():
        modality = source.root / "cleaning_report/source_modality.v21.json"
    if modality.is_file() and not (destination / "meta/modality.json").exists():
        shutil.copy2(modality, destination / "meta/modality.json")
    with OfficialStorage(destination, config) as output:
        if output.info["total_episodes"] != expected_episodes or output.info["total_frames"] != expected_frames:
            raise ValueError("Finalized dataset counts differ from the executed plans")
        cursor = 0
        for index in range(expected_episodes):
            table, row = output.episode(index)
            if int(row["dataset_from_index"]) != cursor:
                raise ValueError("Finalized episode ranges are not contiguous")
            if writer is not None:
                if not np.array_equal(table["frame_index"].to_numpy(), np.arange(len(table))):
                    raise ValueError("Official writer did not rebuild frame_index")
                if not np.allclose(table["timestamp"].to_numpy(), np.arange(len(table))/output.info["fps"], atol=1e-4, rtol=0):
                    raise ValueError("Official writer did not rebuild the uniform timestamp clock")
                if set(output.info["features"]) - {"index", "episode_index", "frame_index", "timestamp", "task_index"} != set(writer.features):
                    raise ValueError("Official writer changed the feature set")
                for key, spec in writer.features.items():
                    actual = output.info["features"][key]
                    if list(actual["shape"]) != list(spec["shape"]) or actual["dtype"] != spec["dtype"]:
                        raise ValueError(f"Output schema differs from the planned transform: {key}")
            cursor += len(table)
        if cursor != expected_frames:
            raise ValueError("Finalized row coverage is incomplete")
        return {"reindex": "official_writer" if writer else "unchanged",
                "metadata": "official_writer" if writer else "preserved",
                "statistics": "official_save_episode" if writer else "preserved_unchanged",
                "episodes": expected_episodes, "frames": expected_frames,
                "lingbot_norm": "recompute with the actual training preprocessing after mutations"}
