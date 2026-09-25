"""LingBot prepare_language compatibility; optional runtime, never imported by core."""
import numpy as np

from lerobot_cleaner.core.language import report_value, text_error
from lerobot_cleaner.training.contracts.lingbot import REVISION, finding


def _tokens(output):
    values = []
    for name in ("input_ids", "attention_mask"):
        value = output[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"{name} must be one batched token sequence")
        values.append(value)
    ids, mask = values
    if ids.shape != mask.shape or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
        raise ValueError("Invalid token IDs or mask shape")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("Attention mask must contain only 0/1")
    return ids, mask.astype(bool)


def evaluate_tokenizer(frequencies, tokenizer, max_length):
    """Real tokenization per unique task, weighted by its actual sample frequency."""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    tasks, findings = [], []
    for text, frequency in sorted(frequencies.items()):
        row = {"text": report_value(text), "samples": frequency, "max_length": max_length,
               "source": "canonical task from LingBot meta.tasks.iloc[task_index].name"}
        try:
            if text_error(text):
                raise ValueError("Task string is invalid before tokenization")
            prompt = text if text.startswith("<bos>") else "<bos>" + text
            prompt = prompt if prompt.endswith("\n") else prompt + "\n"
            # Identical arguments to official prepare_language at the pinned revision.
            actual_ids, actual_mask = _tokens(tokenizer([prompt], padding="max_length",
                padding_side="right", max_length=max_length, truncation=True, return_tensors="pt"))
            full_ids, full_mask = _tokens(tokenizer([prompt], padding=False,
                padding_side="right", truncation=False, return_tensors="pt"))
            if actual_ids.shape[1] != max_length:
                raise ValueError("Tokenizer did not produce the requested padded width")
            raw_count, count = int(full_mask.sum()), int(actual_mask.sum())
            row.update(token_count=raw_count, effective_token_count=count,
                truncated=raw_count > max_length,
                truncation_ratio=max(0, raw_count-count)/raw_count if raw_count else 0.)
            if count != min(raw_count, max_length):
                raise ValueError("Effective token count disagrees with untruncated tokenization")
            if count == 0:
                raise ValueError("Nonempty task produced no effective tokens")
            if actual_mask[0, count:].any() or not actual_mask[0, :count].all():
                raise ValueError("Tokenizer did not apply right padding")
            row["compatible"] = True
            if row["truncated"]:
                findings.append(finding("task_truncated", "Official max_length truncates this task", "WARNING", text=row["text"]))
        except Exception as exc:
            # Tokenizer plugins raise backend-specific exceptions. A failure must be
            # reported, never counted as a passed task or silently skipped.
            row.update(compatible=False, error=report_value(str(exc)))
            findings.append(finding("task_tokenization_error", row["error"], text=row["text"]))
        tasks.append(row)
    completed = [row for row in tasks if row["compatible"]]
    total = sum(row["samples"] for row in completed)
    truncated = sum(row["samples"] for row in completed if row.get("truncated"))
    truncated_tasks = sum(bool(row.get("truncated")) for row in completed)
    if not tasks:
        findings.append(finding("no_tokenizable_tasks", "No valid canonical task strings to tokenize"))
    errors = any(item["severity"] == "ERROR" for item in findings)
    return {"status": "incompatible" if errors else "compatible", "compatible": not errors,
        "max_length": max_length, "padding": "max_length", "padding_side": "right", "truncation": True,
        "unique_tasks_attempted": len(tasks), "unique_tasks_evaluated": len(completed),
        "samples_attempted": sum(row["samples"] for row in tasks), "samples_evaluated": total,
        "truncated_task_count": truncated_tasks,
        "truncated_sample_ratio": truncated/total if total else None,
        "truncated_task_ratio": truncated_tasks/len(completed) if completed else None,
        "ratio_scope": "successfully tokenized tasks only; failures are reported separately",
        "tasks": tasks, "findings": findings, "lingbot_revision": REVISION,
        "scope": "language tokenizer only; not model forward or complete training smoke"}


def check_tokenization(frequencies, contract, options):
    settings = options.tokenization
    if not settings.enabled:
        return {"status": "not_run", "compatible": None, "findings": [],
                "max_length": contract.tokenizer_max_length,
                "reason": "Enable training_check.tokenization.enabled or --tokenize"}
    path = settings.tokenizer_path or contract.provenance.get("tokenizer_path")
    try:
        if not path:
            raise ValueError("Set model.tokenizer_path in training config or tokenization.tokenizer_path")
        from transformers import AutoProcessor
        # Official build_processor uses AutoProcessor, not a replacement tokenizer.
        processor = AutoProcessor.from_pretrained(str(path), padding_side="right",
            trust_remote_code=True, local_files_only=True)
        tokenizer = processor.tokenizer
    except Exception as exc:
        return {"status": "unavailable", "compatible": False,
                "max_length": contract.tokenizer_max_length, "tokenizer_path": str(path),
                "findings": [finding("tokenizer_unavailable", report_value(str(exc)))]}
    report = evaluate_tokenizer(frequencies, tokenizer, contract.tokenizer_max_length)
    report.update(tokenizer_path=str(path), local_files_only=True)
    return report
