"""Pure string/mapping checks. Warnings never change integrity pass/fail."""
from collections import Counter
import unicodedata

from lerobot_cleaner.core.language import report_value, text_error


def check_language(language, *, allow_instruction_changes=False, min_length=2,
                   max_length=4096, check_control_characters=True, check_numeric=True,
                   placeholders=("n/a", "unknown", "task")):
    errors = [dict(issue) for issue in language.issues]
    warnings, evidence = [], []
    # One text validation per source/value, while preserving all associations.
    grouped = {}
    for item in language.evidence + language.samples:
        key = (item.task_source, type(item.task_text).__name__, repr(item.task_text), repr(item.task_index))
        if key not in grouped:
            grouped[key] = {"item": item, "positions": set()}
        if item.sample_index is not None:
            grouped[key]["positions"].add(item.sample_index)
    placeholder_set = {text.casefold() for text in placeholders}
    for entry in grouped.values():
        item = entry["item"]
        row = {"task_index": report_value(item.task_index), "task_text": report_value(item.task_text),
               "task_source": item.task_source, "episode_index": item.episode_index,
               "sample_indices": sorted(entry["positions"])}
        evidence.append(row)
        error = text_error(item.task_text)
        if error:
            errors.append({"code": error, **row})
            continue
        text = item.task_text
        flags = []
        if min_length is not None and len(text) < min_length:
            flags.append("short_task")
        if max_length is not None and len(text) > max_length:
            flags.append("long_task")
        if check_control_characters and any(unicodedata.category(c) in {"Cc", "Cf"} for c in text):
            flags.append("control_characters")
        if check_numeric and text.strip().isnumeric():
            flags.append("numeric_task")
        if text.strip().casefold() in placeholder_set:
            flags.append("placeholder_task")
        warnings.extend({"code": code, "severity": "WARNING", **row} for code in flags)
    frequency = Counter(item.task_text for item in language.samples if text_error(item.task_text) is None)
    if len(frequency) > 1 and not allow_instruction_changes:
        errors.append({"code": "inconsistent_episode_task", "episode_index": language.episode_index})
    # Raw samples carry the original text; the report safely escapes invalid Unicode.
    for error in errors:
        for key, value in list(error.items()):
            if key in {"task_index", "task_text"}:
                error[key] = report_value(value)
        error["severity"] = "ERROR"
    counts = Counter(error["code"] for error in errors)
    has_task = bool(frequency)
    complete = has_task and sum(frequency.values()) == len(language.samples)
    metrics = {"evaluated": True, "episode_index": language.episode_index, "scope": language.scope,
        "has_task": has_task, "complete_task_coverage": complete,
        "samples_total": len(language.samples), "samples_with_task": sum(frequency.values()),
        "unique_task_count": len(frequency), "task_frequency_distribution": dict(frequency),
        "task_text_lengths": {text: len(text) for text in frequency},
        "invalid_task_index_count": sum(counts[c] for c in (
            "invalid_table_index", "duplicate_table_index", "missing_task_index",
            "invalid_task_index_type", "task_index_out_of_range", "unresolved_task_index", "ambiguous_task_index")),
        "empty_task_count": counts["empty_task_text"] + counts["null_task_text"],
        "inconsistent_mapping_count": sum(value for key, value in counts.items() if key.startswith("inconsistent_")),
        "error_counts": dict(counts), "errors": errors, "warnings": warnings, "provenance": evidence}
    if not has_task:
        metrics["errors"].append({"code": "episode_without_task", "severity": "ERROR"})
        metrics["error_counts"]["episode_without_task"] = 1
    return {"rule": "language_integrity", "passed": not metrics["errors"],
            "severity": "error" if metrics["errors"] else "warning" if warnings else "info",
            "message": "Language/task integrity; heuristic warnings are report-only", "metrics": metrics}


def summarize_language(records):
    checks = [row["checks"]["language_integrity"] for row in records if "language_integrity" in row.get("checks", {})]
    metrics = [check["metrics"] for check in checks]
    episodes, samples, errors = Counter(), Counter(), Counter()
    for item in metrics:
        episodes.update(item["task_frequency_distribution"].keys())
        samples.update(item["task_frequency_distribution"])
        errors.update(item["error_counts"])
    with_task = sum(item["has_task"] for item in metrics)
    total = len(metrics)
    return {"status": "disabled" if not checks else "evaluated", "episodes_total": len(records),
        "episodes_evaluated": total, "episodes_with_task": with_task,
        "episodes_without_task": total-with_task,
        "episodes_with_incomplete_task": sum(not item["complete_task_coverage"] for item in metrics),
        "episodes_failed": sum(not check["passed"] for check in checks),
        "missing_ratio": (total-with_task)/total if total else None,
        "task_missing_ratio": (total-with_task)/total if total else None,
        "unique_task_count": len(episodes), "duplicate_task_count": sum(max(n-1, 0) for n in episodes.values()),
        "task_frequency_distribution": dict(episodes), "sample_task_frequency_distribution": dict(samples),
        "frequency_unit": "episodes containing each task; duplicates are INFO only",
        "task_text_lengths": {text: len(text) for text in episodes},
        "invalid_task_index_count": sum(item["invalid_task_index_count"] for item in metrics),
        "empty_task_count": sum(item["empty_task_count"] for item in metrics),
        "inconsistent_mapping_count": sum(item["inconsistent_mapping_count"] for item in metrics),
        "warning_count": sum(len(item["warnings"]) for item in metrics), "error_counts": dict(errors),
        "count_units": "index/mapping issues per sample (table issues per episode); invalid texts per distinct source/value per episode"}
