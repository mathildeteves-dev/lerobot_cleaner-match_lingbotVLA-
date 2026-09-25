"""Official loading -> checks -> plans -> transforms -> v3 writer -> finalizers."""
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import yaml

from lerobot_cleaner.adapters.factory import v3_adapter
from lerobot_cleaner.core.trajectory import _json_safe
from lerobot_cleaner.storage import OfficialStorage, prepare_dataset
from lerobot_cleaner.storage.writer import V3DatasetWriter
from lerobot_cleaner.transforms.executor import execute_plan
from lerobot_cleaner.transforms.episode_filter import EpisodeFilter
from lerobot_cleaner.finalizers import finalize_v3
from .planning import build_plan
from lerobot_cleaner.core.quality.integrity.language import summarize_language
from .percentiles import resolve_percentiles
from .v3_stream_stats import NumericSummary


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _numeric(storage):
    return {key: NumericSummary(int(np.prod(spec["shape"]))) for key, spec in storage.info["features"].items()
            if spec["dtype"] == "bool" or spec["dtype"].startswith(("float", "int", "uint"))}


def _catalog(storage, row, videos):
    for ref in storage.video_references(int(row["episode_index"])):
        relative = ref["path"].relative_to(storage.root).as_posix()
        item = videos.setdefault(relative, {"key": ref["key"], "expected_end": 0.,
                                           "intervals": [], "episode_indices": []})
        item["intervals"].append([ref["start"], ref["end"]])
        item["episode_indices"].append(int(row["episode_index"]))
        item["expected_end"] = max(item["expected_end"], ref["end"])


def _scan(storage, adapter, config, observer=None, output_check=False):
    from lerobot_cleaner.adapters.episode_access import sensor_values
    from .v3_streaming import check_config
    check_config(storage.info, config)
    totals, records, plans, videos = _numeric(storage), [], [], {}
    if observer is not None and hasattr(observer, "set_feature_schema"):
        observer.set_feature_schema(adapter.get_feature_schema())
    frame_count = 0
    image_resources = set()
    unsuccessful = 0
    configured_ids = set(config.transforms.reject_episodes) | set(config.transforms.frame_filter.drop_frames)
    if not output_check and any(index >= storage.info["total_episodes"] for index in configured_ids):
        raise ValueError("Mutation policy references a nonexistent source episode")
    for eid in range(storage.info["total_episodes"]):
        episode = adapter.read_episode(eid)
        storage.validate_episode(episode.metadata, frame_count)
        record, quality_report, plan = build_plan(episode, adapter, config, output_check=output_check)
        records.append(record)
        plans.append(plan)
        if observer is not None:
            observer.process(episode.metadata, episode.df)
        for key, summary in totals.items():
            summary.update(sensor_values(episode.df[key], storage.info["features"][key]["shape"]))
        if "is_episode_successful" in episode.df:
            unsuccessful += int(not bool(episode.df.is_episode_successful.iloc[0]))
        frame_count += len(episode.df)
        _catalog(storage, episode.metadata, videos)
        for key, spec in storage.info["features"].items():
            if spec["dtype"] == "image":
                for value in episode.df[key]:
                    if isinstance(value, dict) and value.get("bytes") is None and value.get("path"):
                        path = storage.path(value["path"])
                        if not path.is_file():
                            raise ValueError(f"Missing referenced image: {path}")
                        image_resources.add(path.relative_to(storage.root).as_posix())
    if frame_count != storage.info["total_frames"]:
        raise ValueError("Official episodes do not cover all input rows")
    files, _, selection = storage.output_layout()
    report = {"dataset": str(storage.root), "version": "v3.0", "adapter": adapter.describe(),
        "engine": "official_episode_pipeline", "requested_engine": config.engine, "episodes": len(records), "frames": frame_count,
        "fps": storage.info["fps"], "tasks": storage.info["total_tasks"],
        "robot_type": storage.info.get("robot_type"), "numeric": {k: v.result() for k, v in totals.items()},
        "language": summarize_language(records),
        "trajectory_quality": records, "transform_plans": [plan.to_dict() for plan in plans],
        "quality_policy": summarize_policy(records), "videos": videos,
        "visual_verification": "disabled" if not (config.quality.enabled and config.quality.visual.enabled) else ("sampled_decode" if config.quality.visual.decode else "metadata_only"),
        "video_verification": "sampled_decode" if config.quality.visual.decode else "metadata_only",
        "data_file_selection": selection, "unsuccessful_episodes": unsuccessful,
        "image_resources": sorted(image_resources),
        "peak_input_batch_rows": storage.peak_rows,
        "peak_episode_rows": max((plan.source_frames for plan in plans), default=0),
        "warning": "Checks do not edit input. Dry-run plans are not executed. Video heuristics are not proof of task quality."}
    if observer is not None:
        report["dataset_review"] = observer.result()
    return report, plans


def summarize_policy(records):
    return {"rejected_episodes": [r["episode_index"] for r in records if not r["quality_report"]["episode_decision"]["keep"]],
            "abort_episodes": [r["episode_index"] for r in records if r["quality_report"]["abort"]],
            "decisions": sum(len(r["quality_report"]["decisions"]) for r in records)}


def _full_video_audit(storage, report, plans, config, *, output_check=False, visual=None, preview_root=None):
    from .v3_streaming import verify_videos
    if visual is not None:
        visual = dict(visual)
        if "preview_episodes" not in visual:
            seen, selected = set(), []
            for row in report.get("dataset_review", {}).get("episode_quality", []):
                if row["task_index"] not in seen and len(selected) < visual.get("preview_limit", 20):
                    seen.add(row["task_index"])
                    selected.append(row["episode_index"])
            visual["preview_episodes"] = selected
    for relative, item in report["videos"].items():
        failure = None
        try:
            verify_videos(storage.root, storage.info, {relative: item}, config,
                          quality=visual, preview_root=preview_root)
        except (OSError, ValueError, RuntimeError) as exc:
            failure = str(exc)
            item["decode_error"] = failure
        for eid in item["episode_indices"]:
            record, plan = report["trajectory_quality"][eid], plans[eid]
            checks = record["checks"]
            check = checks.setdefault("visual_integrity", {"rule": "visual_integrity", "passed": True,
                "severity": "info", "message": None, "metrics": {"evaluated": True}})
            check["metrics"].setdefault("full_decode", {})[relative] = {"passed": failure is None, "error": failure}
            if failure is not None:
                check.update(passed=False, severity="warning", message=failure)
                policy = config.policy.rules.get("visual_integrity", config.policy.default)
                action = policy.on_fail
                decision = {"rule": "visual_integrity", "action": action, "reason": failure}
                quality = record["quality_report"]
                quality["decisions"].append(decision)
                if action == "reject_episode":
                    quality["episode_decision"]["keep"] = False
                    quality["episode_decision"]["reasons"].append("visual_integrity")
                    if not output_check:
                        plan.reject_episode = True
                        plan.reasons.append("visual_integrity")
                if action == "abort" or (output_check and (action == "reject_episode" or config.policy.output_on_fail == "abort")):
                    quality["abort"] = True
            record["quality_report"]["checks"]["visual_integrity"] = _json_safe(check)
            checks["video_integrity"] = {**check, "rule": "video_integrity", "alias_of": "visual_integrity"}
            record["quality_report"]["checks"]["video_integrity"] = _json_safe(checks["video_integrity"])
            record["transform_plan"] = plan.to_dict()
    report["transform_plans"] = [plan.to_dict() for plan in plans]
    report["quality_policy"] = summarize_policy(report["trajectory_quality"])
    report["video_verification"] = "full_decode" if all("decode_error" not in v for v in report["videos"].values()) else "failed"


def audit_pipeline(dataset, config, *, observer=None, video_quality=None, preview_root=None, output_check=False):
    root = prepare_dataset(dataset, config)
    with OfficialStorage(root, config) as storage:
        adapter = v3_adapter(root, config, storage=storage)
        config, references = resolve_percentiles(storage, adapter, config)
        report, plans = _scan(storage, adapter, config, observer, output_check=output_check)
        report["percentile_reference"] = references
        if config.verify_videos:
            _full_video_audit(storage, report, plans, config, output_check=output_check, visual=video_quality, preview_root=preview_root)
        report["dry_run"] = True
        report["dataset_quality"] = {"trajectory_quality": report["trajectory_quality"], "quality_policy": report["quality_policy"], "language": report["language"]}
        report["training_readiness"] = training_report(root, config)
        return report


def clean_pipeline(dataset, output, config, *, resume=False, observer_factory=None,
                   finalize=None, job_metadata=None, video_quality=None):
    from .v3_streaming import fingerprint, job_lock
    original = Path(dataset).resolve()
    root = prepare_dataset(original, config)
    output = Path(output).resolve()
    if output.exists() or any(output.is_relative_to(source) or source.is_relative_to(output) for source in (original, root)):
        raise ValueError("Output must be new and outside source/converted datasets")
    partial = output.with_name(output.name + ".partial")
    if partial.is_symlink() or partial.resolve().is_relative_to(root):
        raise ValueError("Unsafe partial directory")
    if partial.exists() and not resume:
        raise ValueError("Partial job exists; use --resume with unchanged source/config")
    if resume and not partial.exists():
        raise ValueError("No partial job to resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {"format": "transform-plan-v1", "source": str(root), "output": str(output),
                "fingerprint": fingerprint(root), "config": config.model_dump(mode="json"),
                "review": job_metadata}
    partial.mkdir(exist_ok=True)
    with job_lock(partial):
        identity_path = partial / "job.json"
        if resume:
            if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Resume identity differs from source/config; use a new output")
        else:
            _dump(identity_path, identity)
        with OfficialStorage(root, config) as storage:
            adapter = v3_adapter(root, config, storage=storage)
            observer = observer_factory(root) if observer_factory else None
            config, references = resolve_percentiles(storage, adapter, config)
            before, plans = _scan(storage, adapter, config, observer)
            before["percentile_reference"] = references
            if config.verify_videos:
                _full_video_audit(storage, before, plans, config)
            _dump(partial / "input_audit.json", before)
            _dump(partial / "transform_plans.json", [plan.to_dict() for plan in plans])
            if before["quality_policy"]["abort_episodes"]:
                raise ValueError("Input policy aborted cleaning; inspect .partial/input_audit.json")
            retained = EpisodeFilter.select_plans(plans)
            if not retained:
                raise ValueError("All episodes rejected; plans saved, no empty dataset published")
            if fingerprint(root) != identity["fingerprint"]:
                raise ValueError("Source changed after planning")
            estimated = sum(path.stat().st_size for folder in ("meta", "data", "videos")
                            for path in (root/folder).rglob("*") if path.is_file())
            if shutil.disk_usage(output.parent).free < 3 * estimated + int(config.disk_reserve_gb * 1024**3):
                raise ValueError("Insufficient free space for transactional rewrite")
            # An interrupted write restarts from untouched source. Never resume a
            # partially appended official episode or delete a user-owned path.
            with tempfile.TemporaryDirectory(prefix="attempt-", dir=partial) as attempt:
                stage = Path(attempt) / "dataset"
                changes = any(plan.changes_data for plan in plans)
                writer = None
                expected_frames = 0
                applied, changed_values, episode_map = [], {}, []
                with ExitStack() as stack:
                    if changes:
                        writer = V3DatasetWriter(storage, stage, plans)
                        stack.callback(writer.close)
                        for plan in retained:
                            source_episode = adapter.read_episode(plan.episode_id)
                            clean = execute_plan(source_episode, plan, storage.info)
                            writer.write_episode(clean)
                            applied.append({"episode_id": plan.episode_id, "transforms": clean.transform_results})
                            for edit in clean.transform_results["plan"]["metrics"]["numeric"]:
                                key = edit["column"]
                                changed_values[key] = changed_values.get(key, 0) + edit["changed_values"]
                            expected_frames += len(clean.df)
                        episode_map = writer.episode_map
                    else:
                        stage.mkdir()
                        for folder in ("meta", "data", "videos"):
                            if (root/folder).exists():
                                shutil.copytree(root/folder, stage/folder)
                        for relative in before["image_resources"]:
                            target = stage / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists():
                                shutil.copy2(storage.path(relative), target)
                        expected_frames = storage.info["total_frames"]
                        episode_map = [{"source_episode": p.episode_id, "output_episode": p.episode_id} for p in retained]
                    finalized = finalize_v3(writer, storage, stage, config, len(retained), expected_frames)
                # Output checks are observational; no automatic second mutation pass.
                with OfficialStorage(stage, config) as output_storage:
                    output_adapter = v3_adapter(stage, config, storage=output_storage)
                    observer_after = observer_factory(stage) if observer_factory else None
                    after, after_plans = _scan(output_storage, output_adapter, config, observer_after, output_check=True)
                    if config.verify_videos:
                        _full_video_audit(output_storage, after, after_plans, config, output_check=True,
                                          visual=video_quality, preview_root=stage)
                if after["quality_policy"]["abort_episodes"]:
                    _dump(partial / "output_audit.failed.json", after)
                    raise ValueError("Output policy rejected publication; source remains untouched")
                if fingerprint(root) != identity["fingerprint"]:
                    raise ValueError("Source changed during transformations")
                result = {**before, "input": str(root), "output": str(output), "dry_run": False,
                    "episodes": len(retained), "frames": expected_frames,
                    "episodes_removed": len(plans)-len(retained), "rows_removed": before["frames"]-expected_frames,
                    "changed_values": changed_values, "numeric_before": before["numeric"],
                    "numeric_after": after["numeric"], "trajectory_quality_input": before["trajectory_quality"],
                    "trajectory_quality_output": after["trajectory_quality"], "videos": after["videos"],
                    "video_verification": after["video_verification"], "finalizers": finalized,
                    "episode_map": episode_map, "applied_transforms": applied,
                    "quality_policy_output": after["quality_policy"], "tasks": after["tasks"],
                    "video_sha256": {}, "resume_policy": "identity-checked restart from immutable source",
                    "writer": "official_v3_rewrite" if changes else "unchanged_copy"}
                if "dataset_review" in after:
                    result["output_review"] = after["dataset_review"]
                result["dataset_quality"] = {"trajectory_quality": after["trajectory_quality"], "quality_policy": after["quality_policy"], "language": after["language"]}
                result["language"] = after["language"]
                result["language_input"] = before["language"]
                result["language_output"] = after["language"]
                result["training_readiness"] = training_report(stage, config)
                if "dataset" in result["training_readiness"]:
                    result["training_readiness"]["dataset"] = str(output)
                report_dir = stage / "cleaning_report"
                _dump(report_dir / "report.json", result)
                _dump(report_dir / "transform_plans.json", [p.to_dict() for p in plans])
                _dump(report_dir / "model_preprocessing.json", {str(p.episode_id): p.preprocessing for p in plans if p.preprocessing})
                _dump(report_dir / "input_audit.json", before)
                (report_dir / "cleaning_config.used.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
                (report_dir / "report.md").write_text(
                    f"# V3 cleaning\n\nEpisodes: {len(retained)}. Frames: {expected_frames}.\n"
                    f"Removed episodes: {result['episodes_removed']}; frames: {result['rows_removed']}.\n"
                    "Checks produced plans; transforms operated on copies; official output was reopened.\n", encoding="utf-8")
                if finalize:
                    finalize(stage, result)
                _dump(report_dir / "COMPLETE.json", {"status": "complete", "frames": expected_frames})
                if output.exists():
                    raise ValueError("Output appeared during cleaning; refusing overwrite")
                stage.rename(output)
                return result


def training_report(dataset, config):
    """Independent diagnostic: compatibility never rejects or transforms episodes."""
    if config.training_check is None:
        return {"status": "not_requested", "runtime_validated": False}
    from lerobot_cleaner.training.compatibility.lingbot import check_training
    try:
        return check_training(dataset, config.training_check, config)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        return {"status": "unavailable", "compatible": None, "runtime_validated": False,
                "findings": [{"severity": "ERROR", "code": "training_check_unavailable", "message": str(exc)}]}
