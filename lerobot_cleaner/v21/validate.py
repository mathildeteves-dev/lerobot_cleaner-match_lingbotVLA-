"""Input-contract preflight for GR00T-format LeRobot v2.1 datasets.

lerobot-cleaner assumes a specific input shape (see README "Input requirements").
Rather than crash deep inside a rule with a cryptic KeyError, we validate the
contract up front and raise one clear, actionable error listing *everything*
that is wrong, so a user pointing the tool at the wrong kind of dataset learns
why immediately.

The checks are deliberately cheap: meta files + one parquet sample. No video
decoding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lerobot_cleaner.v21.reader import (
    ACTION_COL,
    EPISODES_FILE,
    INFO_FILE,
    META_DIR,
    MODALITY_FILE,
    STATE_COL,
)

#: Standard parquet columns the writer reindexes; all must be present.
REQUIRED_PARQUET_COLUMNS = [
    STATE_COL,
    ACTION_COL,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
]
#: info.json fields the reader/writer rely on.
REQUIRED_INFO_FIELDS = ["data_path", "fps", "chunks_size", "features"]


class InputContractError(ValueError):
    """Raised when the input dataset does not meet the GR00T v2.1 contract."""


def _vector_dim(value) -> int | None:
    """Length of a per-row state/action cell, if it is array-like."""
    try:
        import numpy as np

        arr = np.asarray(value)
        if arr.ndim >= 1:
            return int(arr.shape[-1])
    except Exception:
        pass
    return None


def validate_dataset(root: str | Path) -> list[str]:
    """Check the input contract. Returns a list of warning strings (non-fatal).

    Raises:
        InputContractError: if any hard requirement is violated, with a single
            message enumerating every problem found.
    """
    root = Path(root)
    problems: list[str] = []
    warnings: list[str] = []

    if not root.exists():
        raise InputContractError(f"Dataset root does not exist: {root}")

    meta_dir = root / META_DIR
    if not meta_dir.is_dir():
        raise InputContractError(
            f"No '{META_DIR}/' directory under {root}. "
            f"lerobot-cleaner expects a GR00T-format LeRobot dataset (see README "
            f"'Input requirements')."
        )

    # --- info.json ---
    info = None
    info_path = meta_dir / INFO_FILE
    if not info_path.exists():
        problems.append(f"missing required meta file: meta/{INFO_FILE}")
    else:
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"meta/{INFO_FILE} is not valid JSON: {e}")

    if info is not None:
        for field in REQUIRED_INFO_FIELDS:
            if field not in info:
                problems.append(f"meta/{INFO_FILE} missing field '{field}'")
        version = str(info.get("codebase_version", ""))
        if version and version != "v2.1":
            warnings.append(
                f"codebase_version is '{version}', but this tool targets 'v2.1'. "
                f"It may still work, but the output will be written as v2.1."
            )

    # --- modality.json (GR00T-specific; the load-bearing contract) ---
    modality = None
    modality_path = meta_dir / MODALITY_FILE
    if not modality_path.exists():
        problems.append(
            f"missing required meta file: meta/{MODALITY_FILE}. This file is "
            f"specific to GR00T-format datasets; a plain LeRobot dataset will not "
            f"have it. The GR00T adapter requires it to resolve state/action keys; native v3/LingBot paths do not."
        )
    else:
        try:
            modality = json.loads(modality_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"meta/{MODALITY_FILE} is not valid JSON: {e}")

    if modality is not None:
        for section in ("state", "action"):
            if section not in modality:
                problems.append(f"meta/{MODALITY_FILE} missing '{section}' section")
                continue
            for key, block in modality[section].items():
                if not (isinstance(block, dict) and "start" in block and "end" in block):
                    problems.append(
                        f"meta/{MODALITY_FILE} {section}.{key} must have integer "
                        f"'start' and 'end'"
                    )

    # --- episodes.jsonl + one parquet sample ---
    episodes_path = meta_dir / EPISODES_FILE
    if not episodes_path.exists():
        problems.append(f"missing required meta file: meta/{EPISODES_FILE}")
    elif info is not None and "data_path" in info:
        sample = _first_episode_index(episodes_path)
        if sample is not None:
            chunk = sample // int(info.get("chunks_size", 1000))
            try:
                pq_path = root / info["data_path"].format(
                    episode_chunk=chunk, episode_index=sample
                )
            except (KeyError, IndexError) as e:
                problems.append(f"info.json data_path pattern is malformed: {e}")
                pq_path = None
            if pq_path is not None:
                problems.extend(_check_parquet(pq_path, modality))

    if problems:
        bullet = "\n  - ".join(problems)
        raise InputContractError(
            f"Input dataset at {root} does not meet the GR00T-format LeRobot v2.1 "
            f"contract:\n  - {bullet}\n\nSee the README 'Input requirements' section."
        )
    return warnings


def _first_episode_index(episodes_path: Path) -> int | None:
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    return int(json.loads(line)["episode_index"])
                except (json.JSONDecodeError, KeyError):
                    return None
    return None


def _check_parquet(pq_path: Path, modality: dict | None) -> list[str]:
    problems: list[str] = []
    if not pq_path.exists():
        return [f"first episode parquet not found at expected path: {pq_path}"]
    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:
        return [f"cannot read parquet {pq_path.name}: {e}"]

    missing = [c for c in REQUIRED_PARQUET_COLUMNS if c not in df.columns]
    if missing:
        problems.append(
            f"parquet {pq_path.name} missing required columns: {missing}. "
            f"(state must be column '{STATE_COL}', action must be '{ACTION_COL}'.)"
        )

    # Cross-check modality.json declared dims against the actual vector width.
    if modality is not None and len(df):
        for section, col in (("state", STATE_COL), ("action", ACTION_COL)):
            if col not in df.columns or section not in modality:
                continue
            declared = max((b.get("end", 0) for b in modality[section].values()), default=0)
            actual = _vector_dim(df[col].iloc[0])
            if actual is not None and declared != actual:
                problems.append(
                    f"modality.json declares {section} dim {declared} "
                    f"(max 'end'), but parquet column '{col}' has width {actual}. "
                    f"They must match."
                )
    return problems
