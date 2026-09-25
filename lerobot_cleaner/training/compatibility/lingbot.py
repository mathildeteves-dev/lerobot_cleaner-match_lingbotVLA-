"""Read-only compatibility diagnostics; never creates a TransformPlan."""
from collections import Counter
import numpy as np
import pandas as pd
from lerobot_cleaner.training.contracts.lingbot import finding, load_contract
from .chunks import chunk_statistics, summarize_chunks
from .tokenization import check_tokenization
from lerobot_cleaner.core.language import TaskInfo, text_error, valid_index


def task_text_statistics(frame, tasks):
    counts, valid, frequencies = Counter(), set(), Counter()
    episode_id = int(frame.episode_index.iloc[0]) if len(frame) and "episode_index" in frame else -1
    for position, value in enumerate(frame.get("task_index", pd.Series([None] * len(frame)))):
        if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            counts["missing_task_index"] += 1
            continue
        if not valid_index(value) or value >= len(tasks):
            counts["missing_task_mapping"] += 1
            continue
        row = tasks.iloc[int(value)]
        if not valid_index(row.get("task_index")) or row["task_index"] != value:
            counts["task_mapping_order_mismatch"] += 1
            continue
        # LingBot overrides task using iloc[index].name, not row['task'] or language.
        text = row.name
        if text is None or (isinstance(text, float) and np.isnan(text)) or text is pd.NA:
            counts["null_task_text"] += 1
        elif not isinstance(text, str):
            counts["wrong_type_task_text"] += 1
        elif not text.strip():
            counts["empty_task_text"] += 1
        elif text_error(text):
            counts[text_error(text)] += 1
        else:
            counts["available"] += 1
            valid.add(text)
            canonical = TaskInfo(int(value), text, "task table: LingBot positional index", episode_id, position)
            frequencies[canonical.task_text] += 1
            if "task" in row and (not isinstance(row["task"], str) or row["task"] != text):
                counts["canonical_model_task_mismatch"] += 1
    available = counts["available"]
    return {"samples": len(frame), "available": available, "task_text_available": available == len(frame),
        "task_text_missing_ratio": 1 - available / len(frame) if len(frame) else None,
        "empty_task_count": counts["empty_task_text"], "unique_task_count": len(valid),
        "counts": dict(counts), "task_frequency_distribution": dict(frequencies),
        "canonical_source": "task table index; LingBot positional lookup"}, valid


def camera_availability(storage, frame, episode_id, contract, options):
    targets = {f.target_name: f for f in contract.schema.images}
    result, findings = {}, []
    for target in contract.required_cameras:
        feature = targets.get(target)
        sources = ([feature.origin_key] if feature and feature.origin_key else
                   [s.origin_key for s in feature.sources] if feature else [])
        mask = np.ones(len(frame), dtype=bool) if sources else np.zeros(len(frame), dtype=bool)
        checked, decoded, failures, source_results = 0, 0, [], {}
        for source in dict.fromkeys(sources):
            spec = storage.info["features"].get(source, {})
            available = np.zeros(len(frame), dtype=bool)
            try:
                if spec.get("dtype") == "video":
                    row = storage.episode_metadata(episode_id)
                    prefix = f"videos/{source}/"
                    start, end = float(row[prefix+"from_timestamp"]), float(row[prefix+"to_timestamp"])
                    path = storage.path(storage.meta.get_video_file_path(episode_id, source))
                    timestamps = frame["timestamp"].to_numpy(dtype=float)
                    if path.is_file() and path.stat().st_size and np.isfinite([start, end]).all() and 0 <= start < end:
                        available = np.isfinite(timestamps) & (timestamps >= -1e-6) & (timestamps < end-start-1e-6)
                        if not np.isclose((end-start)*contract.fps, len(frame), atol=.01):
                            failures.append(f"{source}: video interval length differs from episode")
                    if options.decode_cameras:
                        for position in range(0, len(frame), options.camera_sample_stride):
                            checked += 1
                            if not available[position]:
                                continue
                            try:
                                pixels = storage.video_frames(episode_id, source, [timestamps[position]])
                                pixels = pixels.detach().cpu().numpy() if hasattr(pixels, "detach") else np.asarray(pixels)
                                if pixels.ndim != 4 or len(pixels) != 1 or pixels.shape[1] != 3 or not np.isfinite(pixels).all():
                                    raise ValueError("expected one finite RGB CHW frame")
                                decoded += 1
                            except (ValueError, OSError, RuntimeError, AssertionError) as exc:
                                available[position] = False
                                failures.append(f"{source}@{position}: {exc}")
                elif spec.get("dtype") == "image" and source in frame:
                    for position, value in enumerate(frame[source]):
                        present = value is not None
                        if isinstance(value, dict):
                            present = value.get("bytes") is not None or bool(value.get("path") and storage.path(value["path"]).is_file())
                        elif isinstance(value, float) and np.isnan(value):
                            present = False
                        available[position] = present
                        if options.decode_cameras and position % options.camera_sample_stride == 0:
                            checked += 1
                            if present:
                                from lerobot_cleaner.storage.writer import image_array
                                try:
                                    pixels = image_array(value, storage.root)
                                    if pixels.ndim != 3 or pixels.shape[-1] != 3:
                                        raise ValueError("expected RGB image")
                                    decoded += 1
                                except (ValueError, OSError, RuntimeError) as exc:
                                    available[position] = False
                                    failures.append(f"{source}@{position}: {exc}")
                if spec.get("dtype") in {"image", "video"} and (len(spec.get("shape", [])) != 3 or spec["shape"][-1] != 3):
                    failures.append(f"{source}: expected RGB metadata")
                    available[:] = False
            except (KeyError, ValueError, OSError, RuntimeError) as exc:
                failures.append(f"{source}: {exc}")
            mask &= available
            source_results[source] = {"available_frames": int(available.sum()), "missing_frames": int((~available).sum())}
        complete = bool(mask.all()) and bool(sources) and not failures
        result[target] = {"sources": sources, "available_frames": int(mask.sum()), "expected_frames": len(frame),
            "availability_ratio": float(mask.mean()) if len(frame) else None, "complete": complete,
            "checked_decode_frames": checked, "decoded_frames": decoded, "failures": failures,
            "source_coverage": source_results, "scope": "metadata_plus_sampled_decode" if options.decode_cameras else "metadata_and_references_only"}
        if not complete:
            findings.append(finding("camera_unavailable", f"Required camera {target} is incomplete", episode_id=episode_id))
    return result, findings


def evaluate_storage(storage, contract, options):
    findings = list(contract.findings)
    episodes, texts, task_counts = [], set(), Counter()
    task_frequencies = Counter()
    required = contract.required_cameras
    available_targets = [f.target_name for f in contract.schema.images
                         if all(k in storage.info["features"] and storage.info["features"][k]["dtype"] in {"image", "video"}
                                for k in ([f.origin_key] if f.origin_key else [s.origin_key for s in f.sources]))]
    missing = sorted(set(required)-set(available_targets))
    if missing:
        findings.append(finding("required_cameras_missing", "Required training cameras unavailable", missing=missing))
    total_samples = 0
    for episode_id in range(storage.info["total_episodes"]):
        try:
            table, metadata = storage.episode(episode_id)
            frame = table.to_pandas() if hasattr(table, "to_pandas") else table
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            findings.append(finding("episode_unreadable", str(exc), episode_id=episode_id))
            episodes.append({"episode_id": episode_id, "evaluated": False})
            continue
        if not len(frame):
            findings.append(finding("empty_episode", "No training samples", episode_id=episode_id))
            continue
        representations = {}
        for feature in contract.schema.states + contract.schema.actions:
            try:
                values = feature.canonical().extract(frame)
                nonfinite = int((~np.isfinite(values).all(axis=1)).sum())
                representations[feature.target_name] = {"dimension": values.shape[-1], "nonfinite_samples": nonfinite, "compatible": nonfinite == 0}
                if nonfinite:
                    findings.append(finding("nonfinite_mapped_feature", f"{feature.target_name} contains nonfinite training values", episode_id=episode_id))
            except (ValueError, KeyError, TypeError) as exc:
                representations[feature.target_name] = {"compatible": False, "error": str(exc)}
                findings.append(finding("mapped_feature_unavailable", f"{feature.target_name}: {exc}", episode_id=episode_id))
        chunk = chunk_statistics(len(frame), contract.action_chunk_size, options.include_padding_by_timestep)
        task, distinct = task_text_statistics(frame, storage.tasks)
        texts.update(distinct)
        task_counts.update(task["counts"])
        task_frequencies.update(task["task_frequency_distribution"])
        if task["counts"].get("canonical_model_task_mismatch"):
            findings.append(finding("canonical_model_task_mismatch",
                "Generic task column differs from LingBot official table-index text", episode_id=episode_id))
        total_samples += len(frame)
        if not task["task_text_available"]:
            findings.append(finding("task_text_unavailable", "Some samples cannot obtain valid official task text", episode_id=episode_id))
        cameras, errors = camera_availability(storage, frame, episode_id, contract, options)
        findings.extend(errors)
        if options.padding_warning_ratio is not None and chunk["action_chunk_padding_ratio"] > options.padding_warning_ratio:
            findings.append(finding("high_boundary_padding", "Training efficiency diagnostic; episode is retained", "WARNING", episode_id=episode_id))
        episodes.append({"episode_id": episode_id, "evaluated": True, "action_chunk": chunk,
                         "task_text": task, "cameras": cameras, "representations": representations})
    if not total_samples:
        findings.append(finding("no_samples", "No evaluated training samples"))
    camera_slots = sum(c["expected_frames"] for e in episodes for c in e.get("cameras", {}).values())
    available_slots = sum(c["available_frames"] for e in episodes for c in e.get("cameras", {}).values())
    state, action = contract.layout("states"), contract.layout("actions")
    mask_ok = action["compatible"] and len(action["dimension_mask"]) == contract.action_padded_dim and any(action["dimension_mask"])
    if not mask_ok:
        findings.append(finding("invalid_joint_mask", "Action dimension mask cannot match model contract"))
    findings.append(finding("padding_statistics", "Temporal padding is diagnostic, not an episode rejection rule", "INFO"))
    tokenization = check_tokenization(task_frequencies, contract, options)
    findings.extend(tokenization["findings"])
    error = any(f["severity"] == "ERROR" for f in findings)
    return {"target": "lingbot_vla", "status": "incompatible" if error else "static_compatible",
        "compatible": not error, "runtime_validated": False, "training_ready": None if not error else False,
        "tokenization": tokenization,
        "contract": contract.to_dict(), "action_chunk": summarize_chunks(episodes), "episodes": episodes,
        "task_text": {"task_text_available": task_counts["available"] == total_samples and total_samples > 0,
            "task_text_missing_ratio": 1-task_counts["available"]/total_samples if total_samples else None,
            "unique_task_count": len(texts), "empty_task_count": task_counts["empty_task_text"], "counts": dict(task_counts)},
        "cameras": {"required": required, "available": available_targets, "missing": missing,
            "extra": sorted(set(available_targets)-set(required)),
            "extra_dataset_sources": sorted({k for k,v in storage.info["features"].items() if v["dtype"] in {"image", "video"}} -
                {k for f in contract.schema.images for k in ([f.origin_key] if f.origin_key else [s.origin_key for s in f.sources])}),
            "available_dataset_sources": [k for k,v in storage.info["features"].items() if v["dtype"] in {"image", "video"}],
            "camera_availability_ratio": available_slots/camera_slots if camera_slots else None,
            "complete_episodes": sum(bool(e.get("cameras")) and all(c["complete"] for c in e["cameras"].values()) for e in episodes),
            "temporal_validation": "sampled_decode" if options.decode_cameras else "metadata_only"},
        "state": state, "action": action, "joint_mask": {"compatible": mask_ok, "shape": [contract.action_padded_dim],
            "values": action["dimension_mask"], "semantics": "real action dimensions; not state dimensions or time padding"},
        "findings": findings, "mutation_performed": False, "episodes_excluded": [],
        "runtime_requirements": ["matching normalization statistics", "real processor/tokenizer", "official preprocessing + collator smoke"],
        "smoke": {"status": "not_run", "model_required": False, "recommended_level": 2}}


def check_training(dataset, options, loader_config=None):
    from lerobot_cleaner.storage import OfficialStorage
    with OfficialStorage(dataset, loader_config) as storage:
        try:
            contract = load_contract(storage.info, options)
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            return {"target": "lingbot_vla", "status": "incompatible", "compatible": False,
                    "runtime_validated": False, "training_ready": False, "mutation_performed": False,
                    "findings": [finding("invalid_training_contract", str(exc))], "episodes_excluded": []}
        report = evaluate_storage(storage, contract, options)
        report["dataset"] = str(storage.root)
        return report
