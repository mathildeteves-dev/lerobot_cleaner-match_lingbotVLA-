"""End-to-end: clean a synthetic dataset and assert GR00T invariants hold."""

import json
from pathlib import Path

import pandas as pd

from lerobot_cleaner.v21.config import CleaningConfig
from lerobot_cleaner.v21.pipeline import Pipeline
from lerobot_cleaner.v21.video_utils import count_frames


def _read_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def test_end_to_end_alignment(synth_dataset, tmp_path):
    out = tmp_path / "clean"
    cfg = CleaningConfig(
        input=synth_dataset,
        output=out,
        num_workers=2,
        rules={
            "video_integrity": {"enabled": True},
            "timestamp_alignment": {"enabled": True},
            "numeric_sanity": {"enabled": True, "on_nan": "drop_frame"},
            "gripper_binarize": {"enabled": True, "targets": ["state.gripper", "action.gripper"],
                                 "threshold": 0.5},
            "static_frame_trim": {"enabled": True, "mode": "trim_edges",
                                  "pos_threshold": 0.001, "rot_threshold_deg": None},
            "episode_length_filter": {"enabled": True, "min_frames": 30},
        },
    )
    report = Pipeline(cfg).run()

    # ep index 1 (length 12) should be dropped by length filter
    assert 1 in [i for i, _ in report.dropped_episodes]
    assert report.kept_episodes == 2
    assert report.alignment_errors == []

    # meta consistency
    info = json.loads((out / "meta" / "info.json").read_text())
    eps = _read_jsonl(out / "meta" / "episodes.jsonl")
    assert info["total_episodes"] == len(eps) == 2
    assert info["codebase_version"] == "v2.1"

    # per-episode: video frames == parquet rows == episodes.jsonl length, indices reindexed
    total = 0
    for new_idx, rec in enumerate(eps):
        assert rec["episode_index"] == new_idx  # reindexed 0..N
        pq = out / info["data_path"].format(episode_chunk=0, episode_index=new_idx)
        df = pd.read_parquet(pq)
        assert len(df) == rec["length"]
        assert df["frame_index"].tolist() == list(range(len(df)))
        assert df["episode_index"].nunique() == 1 and df["episode_index"].iloc[0] == new_idx
        vpath = out / info["video_path"].format(
            episode_chunk=0, video_key="observation.images.cam", episode_index=new_idx
        )
        assert count_frames(vpath) == len(df)
        total += len(df)
    assert info["total_frames"] == total

    # global index is contiguous across episodes
    all_idx = []
    for new_idx in range(len(eps)):
        pq = out / info["data_path"].format(episode_chunk=0, episode_index=new_idx)
        all_idx += pd.read_parquet(pq)["index"].tolist()
    assert all_idx == list(range(total))

    # gripper binarized
    pq0 = out / info["data_path"].format(episode_chunk=0, episode_index=0)
    import numpy as np
    st = np.stack(pd.read_parquet(pq0)["observation.state"].to_numpy())
    assert set(np.unique(st[:, 3]).tolist()) <= {0.0, 1.0}

    # stats recomputed with real values (not the zero placeholder)
    stats = json.loads((out / "meta" / "stats.json").read_text())
    assert len(stats["observation.state"]["mean"]) == 4
    assert any(abs(v) > 0 for v in stats["observation.state"]["max"])

    # dual-format: episodes_stats.jsonl for upstream-lerobot / pi05
    ep_stats = _read_jsonl(out / "meta" / "episodes_stats.jsonl")
    assert len(ep_stats) == len(eps)
    for new_idx, rec in enumerate(ep_stats):
        assert rec["episode_index"] == new_idx
        s = rec["stats"]
        for key in ("observation.state", "action", "timestamp"):
            assert key in s
            assert set(s[key].keys()) == {"min", "max", "mean", "std", "count"}
        assert s["observation.state"]["count"] == [eps[new_idx]["length"]]
        assert len(s["observation.state"]["mean"]) == 4
        # no video keys (avoids decode; matches GR00T stats.json convention)
        assert not any("image" in k for k in s)

    # report artifacts exist
    assert (out / "cleaning_report" / "report.md").exists()
    assert (out / "cleaning_report" / "cleaning_config.used.yaml").exists()


def test_dry_run_writes_no_data(synth_dataset, tmp_path):
    out = tmp_path / "dry"
    cfg = CleaningConfig(input=synth_dataset, output=out, dry_run=True,
                         rules={"episode_length_filter": {"enabled": True, "min_frames": 30}})
    report = Pipeline(cfg).run()
    assert not (out / "data").exists()
    assert (out / "cleaning_report" / "report.md").exists()
    assert 1 in [i for i, _ in report.dropped_episodes]
