import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot_cleaner.v30.episode_review import task_texts
from lerobot_cleaner.v30.libero import validate_libero
from lerobot_cleaner.v30.v3 import V3Config, clean_v3

PROJECT = Path(__file__).parents[2]


@pytest.fixture
def libero_data(v3_data):
    root = v3_data
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    data_path = root / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(data_path).drop(
        columns=[
            "action.joint_position",
            "action.gripper_position",
            "is_episode_successful",
            "language_instruction",
        ]
    )
    for name in [
        "action.joint_position",
        "action.gripper_position",
        "is_episode_successful",
        "language_instruction",
    ]:
        del info["features"][name]
    state = np.stack(frame["observation.state"])
    frame["observation.states.ee_state"] = list(state[:, :6])
    frame["observation.states.gripper_state"] = list(state[:, 6:8])
    frame["observation.states.joint_state"] = list(np.zeros((len(frame), 7), dtype=np.float32))
    frame["action"] = list(np.stack(frame["action"])[:, :7])
    info["features"]["action"]["shape"] = [7]
    for name, width in [("ee_state", 6), ("gripper_state", 2), ("joint_state", 7)]:
        info["features"][f"observation.states.{name}"] = {"dtype": "float32", "shape": [width]}
    mapping = {"exterior_1_left": "image", "wrist_left": "wrist_image"}
    ep_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(ep_path)
    for old, new in mapping.items():
        old_key, new_key = f"observation.images.{old}", f"observation.images.{new}"
        info["features"][new_key] = info["features"].pop(old_key)
        (root / "videos" / old_key).rename(root / "videos" / new_key)
        episodes.rename(columns={c: c.replace(old_key, new_key) for c in episodes}, inplace=True)
    del info["features"]["observation.images.exterior_2_left"]
    episodes.drop(columns=[c for c in episodes if "exterior_2_left" in c], inplace=True)
    episodes.to_parquet(ep_path, index=False)
    frame.to_parquet(data_path, index=False)
    info_path.write_text(json.dumps(info))
    return root


def test_libero_rejects_droid(v3_data):
    with pytest.raises(ValueError, match="LIBERO schema"):
        validate_libero(v3_data)


def test_indexed_task_texts_and_row_preservation(libero_data, tmp_path):
    result = validate_libero(libero_data, batch_rows=2)
    assert result["frames_without_task_text"] == 0
    assert result["tasks"][0]["text"] == "pick up object"
    assert result["tasks"][0]["episodes"] == 2
    assert result["success_labels_available"] is False
    config = V3Config.from_yaml(PROJECT / "configs/cleaning/libero_v3.yaml")
    config = config.model_copy(
        update={"verify_videos": False, "progress": False, "disk_reserve_gb": 0, "batch_rows": 2}
    )
    output = tmp_path / "clean"
    report = clean_v3(libero_data, output, config)
    assert report["unsuccessful_episodes"] is None
    assert report["changed_values"] == {}
    assert report["rows_removed"] == 0
    pd.testing.assert_frame_equal(
        pd.read_parquet(libero_data / "data/chunk-000/file-000.parquet"),
        pd.read_parquet(output / "data/chunk-000/file-000.parquet"),
    )
    assert (libero_data / "meta/tasks.parquet").read_bytes() == (
        output / "meta/tasks.parquet"
    ).read_bytes()
    assert validate_libero(output)["tasks"] == result["tasks"]


@pytest.mark.parametrize("text", ["", "  ", None, 0])
def test_blank_or_nontext_task_rejected(tmp_path, text):
    path = tmp_path / "tasks.parquet"
    pd.DataFrame({"task_index": [0]}, index=[text]).to_parquet(path)
    with pytest.raises(ValueError, match="task text"):
        task_texts(path)


def test_explicit_task_column(tmp_path):
    path = tmp_path / "tasks.parquet"
    pd.DataFrame({"task_index": [0], "task": ["pick up cup"]}).to_parquet(path, index=False)
    assert task_texts(path) == {0: "pick up cup"}


def test_unresolved_task_rejected(libero_data):
    path = libero_data / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path)
    frame.loc[1, "task_index"] = 8
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="unresolved language"):
        validate_libero(libero_data)


def test_state_semantics_mismatch_rejected(libero_data):
    path = libero_data / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path)
    frame["observation.state"] = [np.zeros(8, dtype=np.float32)] * len(frame)
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="ee_state"):
        validate_libero(libero_data)
