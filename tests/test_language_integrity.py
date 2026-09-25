"""Language evidence and tokenizer boundaries; added without executing per request."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lerobot_cleaner.adapters.builder import AssemblyPolicy, EpisodeBuilder
from lerobot_cleaner.adapters.language import resolve_language, task_catalog
from lerobot_cleaner.adapters.schema import CanonicalFeatureSchema
from lerobot_cleaner.core.quality.integrity.language import check_language, summarize_language
from lerobot_cleaner.training.compatibility.tokenization import evaluate_tokenizer, check_tokenization
from lerobot_cleaner.training.config import TokenizationCheckConfig


def check(frame, catalog=None, metadata=None, **options):
    language = resolve_language(frame, 42, metadata or {}, catalog)
    return check_language(language, **options)


def test_v2_and_v3_catalogs_preserve_text_exactly():
    text = "  pick up the cup\n"
    v2 = task_catalog([{"task_index": 0, "task": text}])
    v3 = task_catalog(pd.DataFrame({"task_index": [0]}, index=[text]))
    assert v2 == v3 == ((0, text),)
    result = check(pd.DataFrame({"task_index": [0, 0]}), v3)
    assert result["passed"]
    assert result["metrics"]["task_frequency_distribution"] == {text: 2}
    assert result["metrics"]["provenance"][0]["task_source"] == "task table"


@pytest.mark.parametrize("value,code", [
    (None, "missing_task_index"), ("0", "invalid_task_index_type"),
    (0.0, "invalid_task_index_type"), (True, "invalid_task_index_type"),
    (-1, "invalid_task_index_type"), (3, "task_index_out_of_range"),
])
def test_invalid_indices_are_reported_even_with_valid_sample_text(value, code):
    frame = pd.DataFrame({"task_index": pd.Series([value], dtype=object), "task": ["pick cup"]})
    result = check(frame, ((0, "pick cup"),), {"task_table_expected_count": 1})
    assert not result["passed"]
    assert result["metrics"]["error_counts"][code] == 1
    assert result["metrics"]["invalid_task_index_count"] == 1


@pytest.mark.parametrize("value,code", [(None, "null_task_text"), ("", "empty_task_text"),
    (" \t", "empty_task_text"), (9, "wrong_type_task_text"), (b"pick", "wrong_type_task_text"),
    ("broken\ud800", "invalid_encoding")])
def test_invalid_text_is_not_repaired_and_report_is_utf8_safe(value, code):
    frame = pd.DataFrame({"task": [value]})
    before = frame.copy(deep=True)
    result = check(frame)
    assert not result["passed"] and result["metrics"]["error_counts"][code] == 1
    json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
    pd.testing.assert_frame_equal(frame, before)


def test_episode_and_sample_evidence_cannot_hide_mapping_conflicts():
    result = check(pd.DataFrame({"task_index": [0], "instruction": ["open drawer"]}),
                   ((0, "pick cup"),), {"tasks": ["close drawer"]})
    assert result["metrics"]["inconsistent_mapping_count"] == 2
    assert {item["task_source"] for item in result["metrics"]["provenance"]} == {
        "episode metadata:tasks", "sample field:instruction", "task table"}


def test_changing_instructions_requires_explicit_opt_in():
    frame = pd.DataFrame({"instruction": ["pick cup", "open drawer"]})
    assert not check(frame)["passed"]
    assert check(frame, allow_instruction_changes=True)["passed"]


def test_episode_only_task_and_equivalent_aliases_need_no_index():
    result = check(pd.DataFrame(index=range(2)), metadata={"task": "pick cup", "instruction": "pick cup"})
    assert result["passed"] and result["metrics"]["scope"] == "episode"
    assert result["metrics"]["samples_with_task"] == 2


def test_unresolved_and_ambiguous_table_indices():
    frame = pd.DataFrame({"task_index": [1]})
    assert check(frame, ((0, "pick cup"),))["metrics"]["error_counts"]["unresolved_task_index"] == 1
    result = check(frame, ((1, "pick cup"), (1, "open drawer")))
    assert not result["passed"]
    assert result["metrics"]["error_counts"]["ambiguous_task_index"] == 1


def test_heuristics_and_repetition_do_not_fail_integrity():
    records = []
    for i in range(3):
        result = check(pd.DataFrame({"task": ["unknown", "unknown"]}))
        assert result["passed"] and result["metrics"]["warnings"]
        records.append({"episode_index": i, "checks": {"language_integrity": result}})
    summary = summarize_language(records)
    assert summary["episodes_total"] == summary["episodes_with_task"] == 3
    assert summary["unique_task_count"] == 1 and summary["duplicate_task_count"] == 2
    assert summary["task_frequency_distribution"] == {"unknown": 3}
    assert summary["sample_task_frequency_distribution"] == {"unknown": 6}
    assert summary["episodes_failed"] == 0
    assert not check(pd.DataFrame(index=range(1)))["passed"]
    assert summarize_language([{"checks": {}}])["status"] == "disabled"


def test_frame_selection_keeps_language_association_and_source_positions():
    frame = pd.DataFrame({"episode_index": [42]*3, "task_index": [0, 1, 0]})
    episode = EpisodeBuilder(CanonicalFeatureSchema(), 10).build(frame,
        task_catalog=((0, "pick cup"), (1, "open drawer")))
    original = episode.language
    episode.restrict_to([0, 2])
    assert [item.sample_index for item in episode.language.samples] == [0, 2]
    assert [item.task_text for item in episode.language.samples] == ["pick cup"]*2
    assert original.samples[1].task_text == "open drawer"


def test_language_stream_alignment_excludes_artificial_padding():
    stream = pd.DataFrame({"timestamp": [0., .2], "task_index": pd.Series([0, 0], dtype=object)})
    builder = EpisodeBuilder(CanonicalFeatureSchema(), 10, AssemblyPolicy(pad_to=4))
    episode = builder.build_streams({"task_index": stream}, [0., .1, .2], episode_id=42,
                                    task_catalog=((0, "pick cup"),))
    result = check_language(episode.language)
    assert result["metrics"]["samples_total"] == 3
    assert result["metrics"]["samples_with_task"] == 2
    assert not result["passed"]  # The real missing middle sample is still visible.


class CharacterTokenizer:
    def __init__(self, empty=False):
        self.empty, self.calls = empty, []

    def __call__(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        count = 0 if self.empty else len(prompts[0])
        if kwargs["truncation"]:
            count = min(count, kwargs["max_length"])
        width = kwargs["max_length"] if kwargs["padding"] == "max_length" else count
        ids = np.zeros((1, width), dtype=np.int64)
        mask = np.zeros_like(ids)
        ids[:, :count], mask[:, :count] = 1, 1
        return {"input_ids": ids, "attention_mask": mask}


def test_tokenizer_receives_official_arguments_and_reports_weighted_truncation():
    tokenizer = CharacterTokenizer()
    report = evaluate_tokenizer({"pick": 2, "a very long task description": 8}, tokenizer, 12)
    assert report["compatible"]
    assert report["truncated_task_ratio"] == .5
    assert report["truncated_sample_ratio"] == .8
    assert all(prompt[0].startswith("<bos>") and prompt[0].endswith("\n") for prompt, _ in tokenizer.calls)
    for _, kwargs in tokenizer.calls[::2]:
        assert kwargs == {"padding": "max_length", "padding_side": "right", "max_length": 12,
                          "truncation": True, "return_tensors": "pt"}


def test_nonempty_task_without_effective_tokens_is_training_error():
    result = evaluate_tokenizer({"pick cup": 3}, CharacterTokenizer(empty=True), 12)
    assert not result["compatible"]
    assert result["findings"][0]["code"] == "task_tokenization_error"
    assert result["findings"][0]["severity"] == "ERROR"


def test_tokenization_disabled_is_not_reported_as_passed():
    report = check_tokenization({"pick": 1}, SimpleNamespace(tokenizer_max_length=48),
                                 SimpleNamespace(tokenization=TokenizationCheckConfig()))
    assert report["status"] == "not_run" and report["compatible"] is None


def test_v3_policy_rejects_integrity_errors_but_not_placeholder_warnings():
    from lerobot_cleaner.v30.planning import build_plan
    from lerobot_cleaner.v30.v3 import V3Config
    config = V3Config(quality={"timestamp": {"enabled": False},
        "episode_structure": False, "video": {"enabled": False}},
        policy={"rules": {"language_integrity": {"on_fail": "reject_episode"}}})
    schema = CanonicalFeatureSchema(raw_vectors=True)
    builder = EpisodeBuilder(schema, 10)
    catalog = ((0, "unknown"),)
    frame = pd.DataFrame({"episode_index": [42]*3, "task_index": [0]*3,
        "timestamp": [0., .1, .2], "observation.state": [[0.], [1.], [2.]],
        "action": [[0.], [1.], [2.]]})
    adapter = SimpleNamespace(info={"features": {
        key: {"dtype": "float32", "shape": [1]} for key in ("observation.state", "action")}},
        from_frame=lambda value: builder.build(value, task_catalog=catalog),
        get_camera_features=lambda: ())
    episode = builder.build(frame, task_catalog=catalog)
    _, report, plan = build_plan(episode, adapter, config)
    assert report.episode_decision.keep and not plan.reject_episode
    frame["task_index"] = 3
    before = frame.copy(deep=True)
    episode = builder.build(frame, task_catalog=catalog)
    record, report, plan = build_plan(episode, adapter, config)
    assert not report.episode_decision.keep and plan.reject_episode
    assert not record["checks"]["language_integrity"]["passed"]
    pd.testing.assert_frame_equal(frame, before)


def test_writer_does_not_coerce_invalid_task_id_or_invalid_unicode():
    from lerobot_cleaner.storage.writer import V3DatasetWriter
    writer = object.__new__(V3DatasetWriter)
    writer.storage = SimpleNamespace(tasks=pd.DataFrame({"task_index": [0]}, index=["pick cup"]))
    assert writer._task(0) == "pick cup"
    for value in (0., "0", True):
        with pytest.raises(ValueError, match="integer task index"):
            writer._task(value)
    writer.storage.tasks.index = ["broken\ud800"]
    with pytest.raises(ValueError, match="task text"):
        writer._task(0)
