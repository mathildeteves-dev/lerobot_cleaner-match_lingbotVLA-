"""Assembly semantics and regressions at the canonical schema boundary."""
import numpy as np
import pandas as pd
import pytest

from lerobot_cleaner.adapters import (
    AssemblyPolicy, CanonicalFeatureSchema, EpisodeBuilder, FeatureSchema, FeatureSlice,
    FeatureResolver,
)


def schema():
    return CanonicalFeatureSchema(
        states=(FeatureSchema("observation.state.arm", "state", (FeatureSlice("q", 0, 1),)),),
        actions=(FeatureSchema("action.arm", "action", (FeatureSlice("u", 0, 1),)),))


def streams():
    return {"q": pd.DataFrame({"timestamp": [0., .1, .2], "q": [[1.], [2.], [3.]]}),
            "u": pd.DataFrame({"timestamp": [.01, .11], "u": [[10.], [20.]]})}


def test_time_pairing_tolerance_and_missing_values():
    builder = EpisodeBuilder(schema(), 10, AssemblyPolicy(alignment="nearest", tolerance=.02))
    episode = builder.build_streams(streams(), [0., .1, .2], episode_id=4)
    view = episode.to_trajectory()
    np.testing.assert_allclose(view.state[:, 0], [1, 2, 3])
    np.testing.assert_allclose(view.action[:, 0], [10, 20, np.nan], equal_nan=True)
    assert episode.metadata["assembly"]["matched"]["u"] == [True, True, False]
    assert episode.feature_schema == schema()


def test_backward_matching_does_not_use_future_action():
    builder = EpisodeBuilder(schema(), 10, AssemblyPolicy(alignment="backward", tolerance=.1))
    view = builder.build_streams(streams(), [0., .1, .2], episode_id=0).to_trajectory()
    np.testing.assert_allclose(view.action[:, 0], [np.nan, 10, 20], equal_nan=True)


@pytest.mark.parametrize("missing,expected", [("preserve", [np.nan, np.nan]),
                                              ("zero", [0, 0]),
                                              ("forward_fill", [np.nan, np.nan])])
def test_exact_matching_and_missing_policy(missing, expected):
    builder = EpisodeBuilder(schema(), 10, AssemblyPolicy(missing=missing))
    view = builder.build_streams(streams(), [0., .1], episode_id=0).to_trajectory()
    np.testing.assert_allclose(view.action[:, 0], expected, equal_nan=True)


def test_forward_fill_preserves_leading_missing_and_never_backfills():
    source = streams()
    source["u"] = pd.DataFrame({"timestamp": [.1], "u": [[20.]]})
    builder = EpisodeBuilder(schema(), 10, AssemblyPolicy(missing="forward_fill"))
    episode = builder.build_streams(source, [0., .1, .2], episode_id=0)
    np.testing.assert_allclose(episode.to_trajectory().action[:, 0], [np.nan, 20, 20], equal_nan=True)
    assert np.isnan(episode.df.u.iloc[-1][0])  # source evidence stays untouched


def test_error_policy_rejects_unmatched_samples():
    builder = EpisodeBuilder(schema(), 10, AssemblyPolicy(missing="error"))
    with pytest.raises(ValueError, match="Missing/nonfinite"):
        builder.build_streams(streams(), [0., .1], episode_id=0).to_trajectory()


@pytest.mark.parametrize("padding,value", [("zero", 0), ("edge", 3), ("nan", np.nan)])
def test_padding_excluded_from_quality_and_explicitly_available(padding, value):
    builder = EpisodeBuilder(schema(), 10, AssemblyPolicy(pad_to=5, padding=padding))
    episode = builder.build_streams(streams(), [0., .1, .2], episode_id=0)
    assert len(episode.df) == 5
    assert len(episode.to_trajectory().state) == 3
    assert episode.metadata["assembly"]["valid_mask"] == [True, True, True, False, False]
    np.testing.assert_allclose(episode.to_trajectory(include_padding=True).state[3:, 0], [value, value], equal_nan=True)
    episode.restrict_to([0, 2, 4])
    np.testing.assert_array_equal(episode.to_trajectory().state[:, 0], [1, 3])


def test_frame_alignment_and_pairing_before_feature_mapping():
    source = {"q": pd.DataFrame({"frame_index": [0, 1, 2], "q": [[1], [2], [3]]}),
              "u": pd.DataFrame({"frame_index": [0, 2], "u": [[8], [9]]})}
    episode = EpisodeBuilder(schema(), 10).build_streams(source, [0, 1, 2], episode_id=1, key="frame_index")
    np.testing.assert_allclose(episode.to_trajectory().timestamps, [0, .1, .2])
    np.testing.assert_allclose(episode.to_trajectory().action[:, 0], [8, np.nan, 9], equal_nan=True)


@pytest.mark.parametrize("timeline", [[0, 0], [1, 0], [0, np.nan]])
def test_invalid_timeline_rejected(timeline):
    with pytest.raises(ValueError, match="strictly increasing"):
        EpisodeBuilder(schema(), 10).build_streams(streams(), timeline, episode_id=0)


def test_cross_episode_stream_rejected():
    source = streams()
    source["q"]["episode_index"] = 8
    with pytest.raises(ValueError, match="another episode"):
        EpisodeBuilder(schema(), 10).build_streams(source, [0., .1], episode_id=0)


def test_matching_requires_bounded_tolerance():
    with pytest.raises(ValueError, match="tolerance"):
        AssemblyPolicy(alignment="nearest")


def test_raw_vector_schema_preserves_gaps_and_live_edits():
    raw = CanonicalFeatureSchema(raw_vectors=True)
    frame = pd.DataFrame({"episode_index": [0], "observation.state": [[1, 2, 3]], "action": [[4, 5, 6]]})
    episode = EpisodeBuilder(raw, 10).build(frame)
    episode.df.at[0, "action"] = [7, 8, 9]
    np.testing.assert_array_equal(episode.to_trajectory().action, [[7, 8, 9]])


def test_resolver_rejects_duplicate_feature_names():
    class Source:
        def get_state_features(self): return schema().states * 2
        def get_action_features(self): return schema().actions
        def get_camera_features(self): return ()
    with pytest.raises(ValueError, match="Duplicate"):
        FeatureResolver().resolve(Source())


def test_missing_whole_vector_and_inconsistent_width():
    frame = pd.DataFrame({"episode_index": [0, 0], "q": [None, [2]], "u": [[3], None]})
    episode = EpisodeBuilder(schema(), 10, AssemblyPolicy(missing="zero")).build(frame)
    np.testing.assert_array_equal(episode.to_trajectory().state[:, 0], [0, 2])
    np.testing.assert_array_equal(episode.to_trajectory().action[:, 0], [3, 0])
    frame.at[0, "q"] = [1, 2]
    with pytest.raises(ValueError, match="Inconsistent vector"):
        episode.to_trajectory()


def test_stream_missing_whole_vector():
    source = streams()
    source["q"].at[1, "q"] = None
    episode = EpisodeBuilder(schema(), 10).build_streams(source, [0., .1], episode_id=0)
    np.testing.assert_allclose(episode.to_trajectory().state[:, 0], [1, np.nan], equal_nan=True)
