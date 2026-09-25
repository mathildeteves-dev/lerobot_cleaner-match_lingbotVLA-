import json

import numpy as np
import pandas as pd
import pytest
from test_v3 import alias_config

from lerobot_cleaner.v30 import v3_streaming as streaming
from lerobot_cleaner.v30.v3 import V3Config, audit_v3, clean_v3, feature_stats
from lerobot_cleaner.v30.v3_stream_stats import OnlineStats


def cfg(**kwargs):
    return alias_config(
        engine="streaming", batch_rows=2, metadata_batch_rows=1, disk_reserve_gb=0, **kwargs
    )


def test_bounded_audit_never_uses_pandas_full_read(v3_data, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Full dataset read/concat is forbidden")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "concat", forbidden)
    report = audit_v3(v3_data, cfg())
    assert report["frames"] == 6
    assert report["peak_input_batch_rows"] <= 2
    assert report["peak_episode_rows"] == 3
    assert report["unsuccessful_episodes"] == 1


def test_streaming_output_matches_memory_engine(v3_data, tmp_path):
    memory, stream = tmp_path / "memory", tmp_path / "stream"
    before = {p.relative_to(v3_data): p.read_bytes() for p in v3_data.rglob("*") if p.is_file()}
    clean_v3(v3_data, memory, alias_config())
    report = clean_v3(v3_data, stream, cfg())
    pd.testing.assert_frame_equal(
        pd.read_parquet(memory / "data/chunk-000/file-000.parquet"),
        pd.read_parquet(stream / "data/chunk-000/file-000.parquet"),
    )
    expected = json.loads((memory / "meta/stats.json").read_text())
    actual = json.loads((stream / "meta/stats.json").read_text())
    for key in expected:
        for stat in expected[key]:
            np.testing.assert_allclose(actual[key][stat], expected[key][stat], atol=1e-14)
    for relative, contents in before.items():
        assert (v3_data / relative).read_bytes() == contents
        if relative.parts[0] == "videos":
            assert (stream / relative).read_bytes() == contents
    assert report["changed_values"] == {}
    assert report["global_quantiles"]["exact_when_all_rows_fit"]
    assert report["video_verification"] == "metadata_only"


def split_files(root):
    path = root / "data/chunk-000/file-000.parquet"
    data = pd.read_parquet(path)
    data.iloc[:2].to_parquet(path, index=False)
    data.iloc[2:5].to_parquet(path.with_name("file-001.parquet"), index=False)
    data.iloc[5:].to_parquet(path.with_name("file-002.parquet"), index=False)
    ep_path = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(ep_path)
    episodes.loc[1, "data/file_index"] = 1
    episodes.iloc[:1].to_parquet(ep_path, index=False)
    episodes.iloc[1:].to_parquet(ep_path.with_name("file-001.parquet"), index=False)
    return data


def test_episode_crosses_batches_and_data_files(v3_data, tmp_path):
    data = split_files(v3_data)
    output = tmp_path / "stream"
    clean_v3(v3_data, output, cfg())
    tables = [pd.read_parquet(p) for p in sorted((output / "data").rglob("*.parquet"))]
    assert [len(t) for t in tables] == [2, 3, 1]
    pd.testing.assert_frame_equal(pd.concat(tables, ignore_index=True), data)
    assert len(list((output / "meta/episodes").rglob("*.parquet"))) == 2
    audit_v3(output, cfg())


def test_interpolation_across_file_boundary_stays_in_episode(v3_data, tmp_path):
    from test_v3 import mutate_actions

    mutate_actions(v3_data, lambda values: values.__setitem__((3, 0), np.nan))
    split_files(v3_data)
    output = tmp_path / "stream"
    clean_v3(v3_data, output, cfg(nonfinite="interpolate"))
    tables = [pd.read_parquet(p) for p in sorted((output / "data").rglob("*.parquet"))]
    data = pd.concat(tables, ignore_index=True)
    assert data.action.iloc[3][0] == pytest.approx(0.32)
    assert data["action.joint_position"].iloc[3][0] == pytest.approx(0.32)


def test_clip_and_aliases(v3_data, tmp_path):
    output = tmp_path / "stream"
    clean_v3(v3_data, output, cfg(bounds={"action": (0, 0.2)}))
    frame = pd.read_parquet(output / "data/chunk-000/file-000.parquet")
    assert np.stack(frame.action).max() == pytest.approx(0.2)
    np.testing.assert_array_equal(np.stack(frame.action)[:, 7], frame["action.gripper_position"])


def test_episode_limit_checked_before_read(v3_data, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Should fail before reading oversized episode")

    monkeypatch.setattr(streaming.FrameCursor, "take", forbidden)
    with pytest.raises(ValueError, match="max_episode_frames"):
        audit_v3(v3_data, cfg(max_episode_frames=2))


def test_footer_count_validated_before_data_read(v3_data, monkeypatch):
    path = v3_data / "meta/info.json"
    info = json.loads(path.read_text())
    info["total_frames"] = 8
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="footer row count"):
        audit_v3(v3_data, cfg())


def test_nonfinite_clean_fails_without_publishing(v3_data, tmp_path):
    from test_v3 import mutate_actions

    mutate_actions(v3_data, lambda values: values.__setitem__((1, 0), np.nan))
    output = tmp_path / "stream"
    with pytest.raises(ValueError, match="Non-finite"):
        clean_v3(v3_data, output, cfg())
    assert not output.exists()
    assert audit_v3(v3_data, cfg())["numeric"]["action"]["nonfinite"] == 1


def test_resume_video_phase_skips_numeric_processing(v3_data, tmp_path, monkeypatch):
    output = tmp_path / "stream"
    original_copy, original_scan = streaming.copy_verified, streaming.scan
    calls = []

    def interrupted(source, target, expected=None):
        calls.append(source)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original_copy(source, target, expected)

    monkeypatch.setattr(streaming, "copy_verified", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        clean_v3(v3_data, output, cfg())
    assert not output.exists()
    assert (tmp_path / "stream.partial/scan.json").is_file()
    assert not (tmp_path / "stream.partial/.lock").exists()
    monkeypatch.setattr(streaming, "copy_verified", original_copy)

    def only_audit(dataset, config, stage=None):
        assert stage is None, "Completed numeric phase should not rerun"
        return original_scan(dataset, config, stage)

    monkeypatch.setattr(streaming, "scan", only_audit)
    clean_v3(v3_data, output, cfg(), resume=True)
    assert output.is_dir()


def test_resume_rejects_changed_source(v3_data, tmp_path, monkeypatch):
    output = tmp_path / "stream"

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupt")

    monkeypatch.setattr(streaming, "copy_verified", interrupted)
    with pytest.raises(RuntimeError):
        clean_v3(v3_data, output, cfg())
    (v3_data / "meta/stats.json").write_text("{} ")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        clean_v3(v3_data, output, cfg(), resume=True)


def test_disk_preflight_before_scan(v3_data, tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(streaming.shutil, "disk_usage", lambda path: usage(10, 10, 0))
    with pytest.raises(ValueError, match="Insufficient disk"):
        clean_v3(v3_data, tmp_path / "stream", cfg())
    assert not (tmp_path / "stream").exists()


def test_lock_prevents_concurrent_resume(v3_data, tmp_path):
    output = tmp_path / "stream"
    partial = tmp_path / "stream.partial"
    partial.mkdir()
    (partial / ".lock").write_text("busy")
    with pytest.raises(ValueError, match="Job locked"):
        clean_v3(v3_data, output, cfg(), resume=True)


def test_stats_match_numpy_and_bound_reservoir():
    values = np.random.default_rng(4).normal(size=(10003, 3))
    stats = OnlineStats(3, capacity=128)
    for start in range(0, len(values), 87):
        stats.update(values[start : start + 87])
    actual = stats.result()
    expected = feature_stats(values)
    for key in ["min", "max", "mean", "std", "count"]:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-13)
    assert stats.sample.shape == (128, 3)
    assert len(np.unique(stats.sample, axis=0)) == 128


def test_stats_stable_for_large_offsets():
    values = 1e10 + np.arange(10000, dtype=float).reshape(-1, 1) / 100
    stats = OnlineStats(1, 16)
    for part in np.array_split(values, 100):
        stats.update(part)
    np.testing.assert_allclose(stats.result()["std"], values.std(axis=0), rtol=1e-6)


def test_global_quantiles_are_labelled_approximate(v3_data, tmp_path):
    report = clean_v3(v3_data, tmp_path / "stream", cfg(quantile_samples=2))
    assert not report["global_quantiles"]["exact_when_all_rows_fit"]
    assert report["global_quantiles"]["samples_per_feature"] == 2


def test_memory_engine_resume_is_rejected(v3_data, tmp_path):
    with pytest.raises(ValueError, match="streaming engine"):
        clean_v3(v3_data, tmp_path / "out", V3Config(), resume=True)


def test_over_million_frames_streams_with_small_batches(v3_data, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = v3_data / "data/chunk-000/file-000.parquet"
    frame = pd.read_parquet(path).iloc[np.arange(512) % 6].reset_index(drop=True)
    frame["frame_index"] = np.arange(512)
    frame["timestamp"] = np.arange(512, dtype=np.float32) / 15
    frame["is_episode_successful"] = True
    ep_path = v3_data / "meta/episodes/chunk-000/file-000.parquet"
    template = pd.read_parquet(ep_path).iloc[0].to_dict()
    episodes = []
    count = 1956
    with pq.ParquetWriter(path, pa.Table.from_pandas(frame, preserve_index=False).schema) as writer:
        for index in range(count):
            frame["episode_index"] = index
            frame["index"] = index * 512 + np.arange(512)
            writer.write_table(pa.Table.from_pandas(frame, preserve_index=False))
            row = {
                **template,
                "episode_index": index,
                "length": 512,
                "dataset_from_index": index * 512,
                "dataset_to_index": (index + 1) * 512,
            }
            for key in row:
                if key.endswith("/from_timestamp"):
                    row[key] = index * 512 / 15
                elif key.endswith("/to_timestamp"):
                    row[key] = (index + 1) * 512 / 15
            episodes.append(row)
    pd.DataFrame(episodes).to_parquet(ep_path, index=False)
    info_path = v3_data / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.update(total_frames=count * 512, total_episodes=count)
    info_path.write_text(json.dumps(info))

    def forbidden(*args, **kwargs):
        raise AssertionError("Full read/concat must not be used for the large dataset")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(pd, "concat", forbidden)
    config = cfg(max_frames=2_000_000)
    config.batch_rows = 2048
    report = audit_v3(v3_data, config)
    assert report["frames"] == 1_001_472
    assert report["peak_input_batch_rows"] <= 2048
    assert report["peak_episode_rows"] == 512


def test_numeric_phase_interruption_restarts_safely(v3_data, tmp_path, monkeypatch):
    output = tmp_path / "out"
    original = streaming.MetadataWriter.append
    calls = 0

    def interrupted(self, *args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("numeric interrupted")
        return original(self, *args)

    monkeypatch.setattr(streaming.MetadataWriter, "append", interrupted)
    with pytest.raises(RuntimeError, match="numeric interrupted"):
        clean_v3(v3_data, output, cfg())
    assert not (tmp_path / "out.partial/scan.json").exists()
    monkeypatch.setattr(streaming.MetadataWriter, "append", original)
    result = clean_v3(v3_data, output, cfg(), resume=True)
    assert result["frames"] == 6


def test_config_change_rejected_on_resume(v3_data, tmp_path, monkeypatch):
    output = tmp_path / "out"

    def interrupted(*args, **kwargs):
        raise RuntimeError("copy interrupted")

    monkeypatch.setattr(streaming, "copy_verified", interrupted)
    with pytest.raises(RuntimeError):
        clean_v3(v3_data, output, cfg())
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        clean_v3(v3_data, output, cfg(quantile_samples=20), resume=True)


def test_bad_completed_checkpoint_rejected(v3_data, tmp_path, monkeypatch):
    output = tmp_path / "out"

    def interrupted(*args, **kwargs):
        raise RuntimeError("copy interrupted")

    monkeypatch.setattr(streaming, "copy_verified", interrupted)
    with pytest.raises(RuntimeError):
        clean_v3(v3_data, output, cfg())
    (tmp_path / "out.partial/dataset/data/chunk-000/file-000.parquet").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checkpoint is damaged"):
        clean_v3(v3_data, output, cfg(), resume=True)


def test_copied_video_corruption_is_repaired_on_resume(v3_data, tmp_path, monkeypatch):
    output = tmp_path / "out"
    original = streaming.copy_verified
    calls = 0

    def interrupted(source, target, expected=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("copy interrupted")
        return original(source, target, expected)

    monkeypatch.setattr(streaming, "copy_verified", interrupted)
    with pytest.raises(RuntimeError):
        clean_v3(v3_data, output, cfg())
    copied = next((tmp_path / "out.partial/dataset/videos").rglob("*.mp4"))
    relative = copied.relative_to(tmp_path / "out.partial/dataset")
    copied.write_bytes(b"corrupted")
    monkeypatch.setattr(streaming, "copy_verified", original)
    clean_v3(v3_data, output, cfg(), resume=True)
    assert (output / relative).read_bytes() == (v3_data / relative).read_bytes()


@pytest.mark.parametrize("frame_count, valid", [(6, True), (5, False)])
def test_online_video_interval_validation(v3_data, monkeypatch, frame_count, valid):
    import sys
    import types
    from fractions import Fraction

    class Container:
        streams = types.SimpleNamespace(video=[types.SimpleNamespace(time_base=Fraction(1, 15))])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            for index in range(frame_count):
                yield types.SimpleNamespace(pts=index, height=180, width=320)

    module = types.ModuleType("av")
    module.open = lambda path: Container()
    monkeypatch.setitem(sys.modules, "av", module)
    if valid:
        report = audit_v3(v3_data, cfg(verify_videos=True))
        assert report["video_verification"] == "full_decode"
        assert all(video["decoded_frames"] == 6 for video in report["videos"].values())
    else:
        with pytest.raises(ValueError, match="frame count mismatch"):
            audit_v3(v3_data, cfg(verify_videos=True))


def test_cli_accepts_streaming_config(v3_data, tmp_path):
    import yaml
    from typer.testing import CliRunner

    from lerobot_cleaner.cli import app

    path = tmp_path / "streaming.yaml"
    path.write_text(yaml.safe_dump(cfg().model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(app, ["audit-v3", str(v3_data), "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert '"engine": "streaming"' in result.output


def test_referenced_policy_excludes_stale_shards_without_deleting(v3_data, tmp_path):
    import shutil

    canonical = v3_data / "data/chunk-000/file-000.parquet"
    stale = canonical.with_name("file-086.parquet")
    shutil.copy2(canonical, stale)
    # Same indices do not imply same values: selection must follow references,
    # not attempt to deduplicate or average conflicting actions.
    table = pd.read_parquet(stale)
    table["action"] = [value + 999 for value in table.action]
    table.to_parquet(stale, index=False)
    stale_bytes = stale.read_bytes()
    with pytest.raises(ValueError, match="footer row count"):
        audit_v3(v3_data, cfg())
    config = cfg(data_file_policy="metadata_referenced")
    audit = audit_v3(v3_data, config)
    assert audit["frames"] == 6
    selection = audit["data_file_selection"]
    assert selection["discovered_files"] == 2
    assert selection["selected_files"] == ["data/chunk-000/file-000.parquet"]
    assert selection["excluded_rows_known"] == 6
    output = tmp_path / "selected"
    clean_v3(v3_data, output, config)
    assert stale.read_bytes() == stale_bytes
    assert not (output / "data/chunk-000/file-086.parquet").exists()
    pd.testing.assert_frame_equal(
        pd.read_parquet(canonical), pd.read_parquet(output / canonical.relative_to(v3_data))
    )
    assert (output / "cleaning_report/data_file_selection.json").is_file()
    # The generated output is also valid under the original strict policy.
    assert audit_v3(output, cfg())["frames"] == 6


def test_referenced_policy_does_not_assume_file_number_cutoff(v3_data):
    import shutil

    canonical = v3_data / "data/chunk-000/file-000.parquet"
    shutil.copy2(canonical, canonical.with_name("file-099.parquet"))
    path = v3_data / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(path)
    episodes["data/file_index"] = 99
    episodes.to_parquet(path, index=False)
    report = audit_v3(v3_data, cfg(data_file_policy="metadata_referenced"))
    assert report["data_file_selection"]["selected_files"] == ["data/chunk-000/file-099.parquet"]


def test_referenced_policy_rejects_missing_referenced_file(v3_data):
    path = v3_data / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pd.read_parquet(path)
    episodes.loc[1, "data/file_index"] = 999
    episodes.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Referenced data files missing"):
        audit_v3(v3_data, cfg(data_file_policy="metadata_referenced"))


def test_referenced_policy_keeps_index_validation(v3_data):
    path = v3_data / "data/chunk-000/file-000.parquet"
    table = pd.read_parquet(path)
    table.loc[4, "index"] = 1
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Invalid index"):
        audit_v3(v3_data, cfg(data_file_policy="metadata_referenced"))


def test_referenced_policy_fails_closed_for_unreferenced_continuation(v3_data):
    # Last file contains an episode continuation, but no episode starts there.
    # Dropping it must fail the coverage check, not publish truncated data.
    split_files(v3_data)
    with pytest.raises(ValueError, match="footer row count"):
        audit_v3(v3_data, cfg(data_file_policy="metadata_referenced"))


def test_referenced_policy_logs_unreadable_unreferenced_file(v3_data):
    (v3_data / "data/chunk-000/file-099.parquet").write_bytes(b"not parquet")
    report = audit_v3(v3_data, cfg(data_file_policy="metadata_referenced"))
    excluded = report["data_file_selection"]["excluded_files"]
    assert len(excluded) == 1
    assert excluded[0]["rows"] is None
    assert "header_error" in excluded[0]


def test_referenced_policy_requires_streaming():
    with pytest.raises(ValueError, match="requires engine=streaming"):
        V3Config(data_file_policy="metadata_referenced")
