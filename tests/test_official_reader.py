"""Contract tests for the official backend; do not run a LingBot runtime."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from lerobot_cleaner.adapters import official
from lerobot_cleaner.adapters.factory import v3_adapter
from lerobot_cleaner.v30.v3 import V3Config, audit_v3, clean_v3


@pytest.fixture
def official_backend(monkeypatch):
    calls = []

    class ArrowDataset:
        def __init__(self, table):
            self.table = table

        def with_format(self, name):
            assert name == "arrow"  # Never torch/video/training transformations.
            return self

        def __len__(self):
            return len(self.table)

        def __getitem__(self, interval):
            return self.table.slice(interval.start, interval.stop - interval.start)

    def open_dataset(root):
        calls.append(root)
        info = json.loads((root / "meta/info.json").read_text())
        frames = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))], ignore_index=True)
        episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("*/*.parquet"))], ignore_index=True)
        return SimpleNamespace(hf_dataset=ArrowDataset(pa.Table.from_pandas(frames, preserve_index=False)),
                               meta=SimpleNamespace(info=info, episodes=episodes.to_dict("records")))

    monkeypatch.setattr(official, "open_official_dataset", open_dataset)
    return calls


@pytest.mark.parametrize("engine", ["memory", "streaming"])
def test_official_backend_drives_audit_clean_and_output_audit(v3_data, tmp_path, official_backend, engine):
    config = V3Config(reader_backend="lerobot", engine=engine, disk_reserve_gb=0)
    report = audit_v3(v3_data, config)
    assert report["adapter"]["reader_backend"] == "lerobot"
    output = tmp_path / "official-output"
    clean_v3(v3_data, output, config)
    assert v3_data in official_backend
    assert any(root != v3_data for root in official_backend)  # Output is read again.
    for path in (v3_data / "data").glob("*/*.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(path), pd.read_parquet(output / path.relative_to(v3_data)))


def test_official_episode_uses_metadata_and_limits(v3_data, official_backend):
    config = V3Config(reader_backend="lerobot")
    adapter = v3_adapter(v3_data, config)
    episode = adapter.read_episode(1)
    assert episode.episode_index == 1 and len(episode.df) == 3
    assert [ep.episode_index for ep in adapter.iter_episodes()] == [0, 1]
    with pytest.raises(KeyError, match="Unknown episode"):
        adapter.read_episode(2)
    config.max_episode_frames = 1
    with pytest.raises(ValueError, match="frame limit"):
        adapter.read_episode(0)


def test_no_silent_native_fallback(v3_data, monkeypatch):
    def fail(root):
        raise RuntimeError("official dependency unavailable")
    monkeypatch.setattr(official, "open_official_dataset", fail)
    for engine in ("memory", "streaming"):
        with pytest.raises(RuntimeError, match="official dependency unavailable"):
            audit_v3(v3_data, V3Config(reader_backend="lerobot", engine=engine))


def test_official_rejects_recovery_file_policy():
    with pytest.raises(ValueError, match="strict data_file_policy"):
        V3Config(reader_backend="lerobot", engine="streaming", data_file_policy="metadata_referenced")


def test_real_import_boundary_uses_official_class_without_download(v3_data, monkeypatch):
    module = ModuleType("lerobot.datasets.lerobot_dataset")

    class FakeLeRobotDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_hf_dataset(self):
            raise FileNotFoundError("missing shard")

    module.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", ModuleType("lerobot"))
    monkeypatch.setitem(sys.modules, "lerobot.datasets", ModuleType("lerobot.datasets"))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    dataset = official.open_official_dataset(v3_data)
    assert isinstance(dataset, FakeLeRobotDataset)
    assert dataset.kwargs["root"] == v3_data.resolve()
    assert dataset.kwargs["download_videos"] is False
    assert dataset.kwargs["revision"] == "local"
    with pytest.raises(RuntimeError, match="download is disabled"):
        dataset.download()
    with pytest.raises(RuntimeError, match="rejected local data"):
        dataset.load_hf_dataset()


def test_official_rejects_reordered_data(v3_data, official_backend, monkeypatch):
    reader = official.OfficialReader(v3_data)
    reader.table.table = reader.table.table.take(pa.array(list(reversed(range(len(reader.table))))))
    with pytest.raises(ValueError, match="order/global indices"):
        reader.episode(0)
    reader.close()


def test_shipped_backend_choices():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "configs/cleaning"
    assert V3Config.from_yaml(root / "droid_v3.yaml").reader_backend == "lerobot"
    assert V3Config.from_yaml(root / "droid_v3_referenced.yaml").reader_backend == "native"
