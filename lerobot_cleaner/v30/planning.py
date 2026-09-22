"""Checks -> policy decisions -> plans. This module never edits an episode."""

import numpy as np

from lerobot_cleaner.adapters.schema import FeatureSlice, vector_values
from lerobot_cleaner.core.plans import EpisodeDecision, NumericEdit, QualityReport, TransformPlan
from lerobot_cleaner.core.quality._common import result
from lerobot_cleaner.core.quality.motion.static_edges import check_static_edges
from .visual import visual_findings
from .quality import audit_trajectory

from lerobot_cleaner.adapters.episode_access import IDENTITY_COLUMNS, sensor_values, episode_identity


def resolve_target(adapter, name):
    schema = adapter.get_feature_schema()
    for feature in schema.states + schema.actions:
        if feature.name == name:
            return feature.slices
    spec = adapter.info["features"].get(name)
    if spec is not None and spec["dtype"].startswith("float") and len(spec["shape"]) == 1:
        return (FeatureSlice(name, 0, spec["shape"][0]),)
    raise ValueError(f"Unknown numeric transform target: {name}")


def numeric_edits(adapter, target, operation, parameters):
    slices = resolve_target(adapter, target)
    width = sum(part.end - part.start for part in slices)
    parameters = dict(parameters)
    if operation in {"clip", "percentile_clip"}:
        low, high = np.asarray(parameters["low"], dtype=float), np.asarray(parameters["high"], dtype=float)
        if (low.ndim > 1 or high.ndim > 1 or low.size not in {1, width} or high.size not in {1, width}
                or not np.isfinite(low).all() or not np.isfinite(high).all()):
            raise ValueError(f"Invalid clipping reference bounds: {target}")
        low, high = np.broadcast_to(low, (width,)), np.broadcast_to(high, (width,))
        if np.any(low > high):
            raise ValueError(f"Unordered clipping reference bounds: {target}")
    offset, edits = 0, []
    for part in slices:
        if part.column in IDENTITY_COLUMNS:
            raise ValueError("Numeric transforms cannot modify identity/time fields")
        spec = adapter.info["features"].get(part.column, {})
        if not spec.get("dtype", "").startswith("float"):
            raise ValueError("Numeric transforms require floating-point storage columns")
        params = dict(parameters)
        size = part.end - part.start
        if operation in {"clip", "percentile_clip"}:
            params.update(low=low[offset:offset+size].tolist(), high=high[offset:offset+size].tolist())
        edits.append(NumericEdit(part.column, part.start, part.end, operation, params))
        offset += size
    return edits


def video_findings(storage, episode, options):
    """Compatibility entry point; both image and video now use canonical visuals."""
    from lerobot_cleaner.adapters.resolver import FeatureResolver
    from lerobot_cleaner.adapters.schema import FeatureSchema
    features = (episode.feature_schema.visual if episode.feature_schema is not None else
                tuple(FeatureResolver().resolve_visual(FeatureSchema(key, "visual", camera_column=key), storage.info["features"])
                      for key, spec in storage.info["features"].items() if spec.get("dtype") in {"image", "video"}))
    return visual_findings(storage, episode, options, features)


def findings(record):
    yield from ((key, check) for key, check in record["checks"].items() if not check.get("alias_of"))
    for group, details in record.get("groups", {}).items():
        for rule, check in details["checks"].items():
            yield f"groups/{group}/{rule}", check


def build_plan(episode, adapter, config, *, output_check=False):
    """Return old-compatible metric records plus explicit report and mutation plan."""
    frame = episode.df
    eid = episode.episode_index
    record = (audit_trajectory(frame, episode.fps, config.quality, adapter=adapter,
                              metadata=episode.metadata) if config.quality.enabled else
              {"episode_index": eid, "checks": {}, "mode": "disabled"})
    if config.quality.enabled and config.quality.language.enabled:
        from lerobot_cleaner.core.quality.integrity.language import check_language
        record["checks"]["language_integrity"] = check_language(
            episode.language, **config.quality.language.model_dump(exclude={"enabled"}))
    plan = TransformPlan(eid, len(frame), episode_identity(episode))
    mutations = config.transforms
    # Find bad values across all declared numeric sensor columns, not just state/action.
    bad_rows = np.zeros(len(frame), dtype=bool)
    bad_columns = {}
    for column, spec in adapter.info["features"].items():
        if not spec["dtype"].startswith("float"):
            continue
        arr = sensor_values(frame[column], spec["shape"])
        mask = ~np.isfinite(arr).all(axis=1)
        bad_rows |= mask
        if mask.any():
            bad_columns[column] = np.flatnonzero(mask).tolist()
            if not output_check and config.nonfinite == "interpolate" and column not in IDENTITY_COLUMNS:
                plan.numeric.append(NumericEdit(column, 0, arr.shape[1], "interpolate"))
    finite = record["checks"].setdefault("finite", result("finite", not bad_columns, {}).to_dict())
    finite["passed"] = finite["passed"] and not bad_columns
    finite["metrics"].update(evaluated=True, bad_source_frames=np.flatnonzero(bad_rows).tolist(),
                             nonfinite_columns=bad_columns)
    if config.quality.enabled and config.quality.visual.enabled:
        record["checks"].update(visual_findings(adapter.storage, episode, config.quality.visual, adapter.get_visual_features()))
    static = mutations.static_trim
    if static.enabled:
        view = episode.to_trajectory()
        if static.columns is not None and max(static.columns) >= getattr(view, static.source).shape[1]:
            raise ValueError("Static selection exceeds canonical dimensions")
        check = check_static_edges(view, static.source, static.columns, static.epsilon, static.min_kept_frames)
        record["checks"]["static_edges"] = check.to_dict()
        if not output_check and check.metrics.get("evaluated"):
            if static.mode == "trim_edges" and check.metrics.get("proposal") is not None:
                plan.trim = tuple(check.metrics["proposal"])
            elif static.mode == "drop_static_frames":
                plan.drop_frames.extend(check.metrics.get("drop_frames", []))
    if not output_check:
        if mutations.frame_filter.nonfinite:
            plan.drop_frames.extend(np.flatnonzero(bad_rows).tolist())
        plan.drop_frames.extend(mutations.frame_filter.drop_frames.get(eid, []))
        if any(position >= len(frame) for position in plan.drop_frames):
            raise ValueError(f"drop_frames index outside episode {eid}")
        plan.drop_frames = sorted(set(plan.drop_frames))
        for target, bounds in config.bounds.items():
            plan.numeric.extend(numeric_edits(adapter, target, "clip", {"low": bounds[0], "high": bounds[1]}))
        for item in mutations.numeric:
            params = item.model_dump(exclude={"target", "operation"}, exclude_none=True)
            plan.numeric.extend(numeric_edits(adapter, item.target, item.operation, params))
        plan.reindex = mutations.retime or plan.trim is not None or bool(plan.drop_frames)
        cameras = {f.name: f.camera_column for f in adapter.get_camera_features()}
        for target, crop in mutations.crop.items():
            column = cameras.get(target, target)
            if column is None:
                raise ValueError(f"ROI target {target!r} is a derived image; select a physical source camera for dataset cropping")
            spec = adapter.info["features"].get(column, {})
            if spec.get("dtype") not in {"video", "image"}:
                raise ValueError(f"ROI target is not an image/video: {target}")
            height, width = spec["shape"][:2]
            if crop.box[2] > width or crop.box[3] > height:
                raise ValueError(f"ROI outside {target} dimensions")
            destination = plan.crop if crop.purpose == "dataset" else plan.preprocessing
            destination[column] = crop.model_dump()
        if mutations.crop:
            record["checks"]["roi"] = result("roi", True, {"evaluated": True,
                "dataset": plan.crop, "preprocessing": plan.preprocessing}).to_dict()
    alias_failures = []
    for alias in config.aliases:
        source = sensor_values(frame[alias.source], adapter.info["features"][alias.source]["shape"])
        target = sensor_values(frame[alias.target], adapter.info["features"][alias.target]["shape"])
        if not np.allclose(source[:, alias.start:alias.end], target, rtol=1e-5, atol=1e-8, equal_nan=True):
            alias_failures.append(alias.target)
        if not output_check and (plan.numeric or alias.target in alias_failures):
            if alias.target in IDENTITY_COLUMNS:
                raise ValueError("Aliases cannot overwrite identity columns")
            plan.numeric.append(NumericEdit(alias.target, 0, target.shape[1], "alias",
                {"source": alias.source, "start": alias.start, "end": alias.end}))
    if config.aliases:
        record["checks"]["aliases"] = result("aliases", not alias_failures,
            {"evaluated": True, "mismatched_targets": alias_failures}).to_dict()
    if output_check and mutations.crop:
        record["checks"]["roi"] = result("roi", True,
            {"evaluated": True, "scope": "output shape validated by writer/finalizer"}).to_dict()
    report = QualityReport(eid, record["checks"], record.get("groups", {}))
    available = dict(findings(record))
    for key in config.policy.rules:
        if key not in available and not any(path.rsplit("/", 1)[-1] == key for path in available):
            raise ValueError(f"Policy {key} has no enabled check; configure its quality settings first")
    for path, check in available.items():
        policy = config.policy.rules.get(path, config.policy.rules.get(path.rsplit("/", 1)[-1], config.policy.default))
        missing = check.get("metrics", {}).get("evaluated") is False
        if check["passed"] and not missing:
            continue
        action = policy.on_unevaluated if missing else policy.on_fail
        report.decisions.append({"rule": path, "action": action, "unevaluated": missing})
        if action == "abort":
            report.abort = True
        if action == "reject_episode":
            if output_check:
                report.abort = True
            report.episode_decision.keep = False
            report.episode_decision.reasons.append(path)
    if not output_check:
        if eid in mutations.reject_episodes:
            report.episode_decision.keep = False
            report.episode_decision.reasons.append("explicit_episode_filter")
        keep = np.ones(len(frame), dtype=bool)
        if plan.trim is not None:
            keep[:plan.trim[0]] = False
            keep[plan.trim[1]:] = False
        keep[plan.drop_frames] = False
        minimum = max(mutations.min_kept_frames, static.min_kept_frames if static.enabled else 1)
        if int(keep.sum()) < minimum:
            if mutations.on_too_short == "abort":
                report.abort = True
            else:
                report.episode_decision.keep = False
                report.episode_decision.reasons.append("too_short_after_frame_selection")
        plan.reject_episode = not report.episode_decision.keep
        plan.reasons = list(report.episode_decision.reasons)
        # Never hand non-finite unrepairable values to an output writer.
        for column, positions in bad_columns.items():
            unhandled = any(keep[position] for position in positions) and not plan.reject_episode
            repaired = any(edit.column == column and edit.operation == "interpolate" for edit in plan.numeric)
            if unhandled and not repaired:
                report.abort = True
                report.decisions.append({"rule": f"finite/{column}", "action": "abort",
                                         "reason": "remaining nonfinite values have no repair plan"})
    if output_check and config.policy.output_on_fail == "abort" and report.decisions:
        report.abort = True
    record.update(quality_report=report.to_dict(), transform_plan=plan.to_dict())
    return record, report, plan
