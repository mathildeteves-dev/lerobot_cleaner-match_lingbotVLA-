import numpy as np
import pandas as pd

from lerobot_cleaner.v21.config import (
    GripperBinarizeConfig,
    NumericSanityConfig,
    OnBad,
    StaticFrameTrimConfig,
    StaticTrimMode,
)
from lerobot_cleaner.v21.reader import ModalityResolver
from lerobot_cleaner.v21.rules.checks.integrity.numeric import NumericSanityRule
from lerobot_cleaner.v21.rules.transforms.embodiment.gripper import GripperBinarizeRule
from lerobot_cleaner.v21.rules.transforms.motion.static_trim import StaticFrameTrimRule
from lerobot_cleaner.v21.types import EpisodeWork


class FakeRef:
    def __init__(self):
        self.episode_index = 0
        self.tasks = ["t"]
        self.video_paths = {}


class FakeDataset:
    def __init__(self):
        self.resolver = ModalityResolver({
            "state": {"arm": {"start": 0, "end": 3}, "gripper": {"start": 3, "end": 4}},
            "action": {"arm": {"start": 0, "end": 3}, "gripper": {"start": 3, "end": 4}},
        })
        self.fps = 10.0


def _work(state, action=None):
    if action is None:
        action = state.copy()
    df = pd.DataFrame({
        "observation.state": list(state.astype(np.float32)),
        "action": list(action.astype(np.float32)),
        "timestamp": (np.arange(len(state)) / 10.0).astype(np.float32),
    })
    return EpisodeWork(ref=FakeRef(), df=df, keep_indices=list(range(len(state))))


def test_gripper_binarize_and_idempotent():
    ds = FakeDataset()
    state = np.zeros((5, 4), dtype=np.float32)
    state[:, 3] = [0.1, 0.8, 0.4, 0.9, 0.2]
    work = _work(state)
    cfg = GripperBinarizeConfig(enabled=True, targets=["state.gripper"], threshold=0.5)
    rule = GripperBinarizeRule(cfg, ds)
    rule.apply(work)
    out = np.stack(work.df["observation.state"].to_numpy())
    assert set(np.unique(out[:, 3]).tolist()) <= {0.0, 1.0}
    assert out[:, 3].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
    # second pass is a no-op (idempotent)
    rule2 = GripperBinarizeRule(cfg, ds)
    rule2.apply(work)
    assert rule2.stats.get("skipped_already_binary:state.gripper", 0) == 1


def test_static_trim_edges_preserves_interior():
    ds = FakeDataset()
    state = np.zeros((20, 4), dtype=np.float32)
    state[:, 0] = np.concatenate([np.zeros(5), np.linspace(0, 1, 10), np.ones(5)])
    work = _work(state)
    cfg = StaticFrameTrimConfig(enabled=True, mode=StaticTrimMode.trim_edges,
                                pos_threshold=1e-3, rot_threshold_deg=None)
    rule = StaticFrameTrimRule(cfg, ds)
    n0 = len(work.df)
    rule.apply(work)
    assert len(work.df) < n0
    # keep_indices stays consistent with df
    assert len(work.keep_indices) == len(work.df)
    # interior is contiguous (edges only trimmed)
    assert work.keep_indices == list(range(work.keep_indices[0], work.keep_indices[-1] + 1))


def test_numeric_nan_drop_episode():
    ds = FakeDataset()
    state = np.zeros((10, 4), dtype=np.float32)
    state[3, 0] = np.nan
    work = _work(state)
    cfg = NumericSanityConfig(enabled=True, on_nan=OnBad.drop_episode)
    rule = NumericSanityRule(cfg, ds)
    rule.apply(work)
    assert work.dropped


def test_numeric_outlier_clip():
    ds = FakeDataset()
    state = np.zeros((100, 4), dtype=np.float32)
    state[:, 0] = np.linspace(-1, 1, 100)
    state[0, 0] = 999.0  # outlier
    work = _work(state)
    cfg = NumericSanityConfig(enabled=True, outlier_mode="clip_quantile")
    bounds = {"state": (np.array([-1, -1, -1, -1.0]), np.array([1, 1, 1, 1.0])),
              "action": (np.array([-1, -1, -1, -1.0]), np.array([1, 1, 1, 1.0]))}
    rule = NumericSanityRule(cfg, ds, outlier_bounds=bounds)
    rule.apply(work)
    out = np.stack(work.df["observation.state"].to_numpy())
    assert out[:, 0].max() <= 1.0 + 1e-6
