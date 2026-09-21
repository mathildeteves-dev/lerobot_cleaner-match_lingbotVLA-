"""Read-only per-group calibration and optional conservative v3 cleaning."""
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lerobot_cleaner.core.quality.calibration import robust_threshold
from lerobot_cleaner.v30.v3 import V3Config, audit_v3, clean_v3
from lerobot_cleaner.v30.v3_streaming import fingerprint

METRICS = {
    "velocity": "max_velocity",
    "acceleration": "max_acceleration",
    "jerk": "max_jerk",
    "velocity_zscore": "max_zscore",
    "acceleration_zscore": "max_zscore",
}


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    quantile: float = Field(0.995, gt=0, lt=1)
    mad_k: float = Field(8.0, ge=0)
    mad_scale: float = Field(1.4826, gt=0)
    min_samples: int = Field(20, ge=2, strict=True)
    positive_floor: float = Field(1e-12, gt=0)


def estimate_groups(records, quality, settings):
    """Calibrate every metric separately, including checks that previously failed."""
    if not quality.groups:
        raise ValueError("Calibration requires explicit quality.groups; no robot layout is inferred")
    grouped = {}
    complete = True
    for group in quality.groups:
        metrics = {}
        for rule, measurement in METRICS.items():
            values = []
            excluded = Counter()
            for record in records:
                check = record.get("groups", {}).get(group.name, {}).get("checks", {}).get(rule)
                if check is None:
                    excluded["missing_check"] += 1
                elif not check["metrics"].get("evaluated", False):
                    excluded[check.get("message") or "not_evaluated"] += 1
                else:
                    value = check["metrics"].get(measurement)
                    values.append(float("nan") if value is None else value)
            result = robust_threshold(values, **settings.model_dump())
            if result["excluded"]:
                excluded["invalid_measurement"] += result["excluded"]
            result.update(measurement=measurement, total_episodes=len(records),
                          excluded_episodes=sum(excluded.values()), exclusion_reasons=dict(excluded))
            metrics[rule] = result
            complete &= result["status"] == "ready"
        grouped[group.name] = {"source": group.source, "columns": group.columns, "metrics": metrics}
    return grouped, bool(complete)


def _outside_new(path, root):
    if path.exists() or path.is_relative_to(root) or root.is_relative_to(path):
        raise ValueError("Output must be a new directory outside the input dataset")


def calibrate_v3(dataset, output, config, settings=None, *, clean_output=None):
    """Publish report + full reusable V3Config, then optionally run clean-v3.

    Statistics use exact per-episode maxima, equally weighted. Streaming bounds
    frame memory, while collected episode summaries scale with episode count.
    """
    settings = settings or CalibrationConfig()
    if not config.quality.enabled or not config.quality.groups:
        raise ValueError("Calibration requires quality.enabled=true and explicit quality.groups")
    _outside_new(Path(output).resolve(), Path(dataset).resolve())
    if clean_output is not None:
        _outside_new(Path(clean_output).resolve(), Path(dataset).resolve())
    from lerobot_cleaner.storage import prepare_dataset
    root, output = prepare_dataset(dataset, config), Path(output).resolve()
    _outside_new(output, root)
    if clean_output is not None:
        clean_output = Path(clean_output).resolve()
        _outside_new(clean_output, root)
        if output.is_relative_to(clean_output) or clean_output.is_relative_to(output):
            raise ValueError("Calibration and cleaning outputs must be separate directories")
    source_identity = fingerprint(root)
    # audit_v3 never repairs, drops, trims, or clips, regardless of cleaning settings.
    audit = audit_v3(root, config)
    if fingerprint(root) != source_identity:
        raise ValueError("Source changed during calibration; keep input immutable")
    records = audit["trajectory_quality"]
    groups, ready = estimate_groups(records, config.quality, settings)
    report = {
        "schema_version": 1, "status": "ready" if ready else "insufficient_calibration",
        "dataset": str(root), "episodes": audit["episodes"], "frames": audit["frames"],
        "source_fingerprint": source_identity,
        "fingerprint_method": "info content plus path/size/mtime; not a full content hash",
        "population": "one finite peak per episode per feature group; equal episode weighting",
        "quantile_method": "exact_linear", "formula": "max(quantile, median + mad_k * mad_scale * MAD)",
        "parameters": settings.model_dump(), "groups": groups,
        "input_config": config.model_dump(mode="json"), "warnings": [],
        "threshold_config": None, "cleaning": {"status": "not_requested" if clean_output is None else "blocked"},
        "policy": "Read-only calibration; generated thresholds only affect quality checks, not row removal or numeric bounds.",
    }
    for name, group in groups.items():
        for metric, stats in group["metrics"].items():
            if stats["excluded_episodes"]:
                report["warnings"].append(f"{name}/{metric}: {stats['excluded_episodes']} episodes excluded; see reasons")
            if stats["status"] == "ready" and stats["expected_upper_tail_samples"] < 1:
                report["warnings"].append(f"{name}/{metric}: fewer than one expected upper-tail sample; quantile estimate is uncertain")
            if stats.get("floor_applied"):
                report["warnings"].append(f"{name}/{metric}: applied explicit positive threshold floor")
    generated = None
    if ready:
        generated_dict = config.model_dump(mode="json")
        for group in generated_dict["quality"]["groups"]:
            for metric, stats in groups[group["name"]]["metrics"].items():
                group[metric] = stats["threshold"]
        generated = V3Config.model_validate(generated_dict)
        report["threshold_config"] = str(output / "thresholds.yaml")
    # Publish only complete artifacts; never overwrite an earlier calibration.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibration-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        (stage / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if generated is not None:
            contents = yaml.safe_dump(generated.model_dump(mode="json"), sort_keys=False)
            (stage / "thresholds.yaml").write_text(contents, encoding="utf-8")
            report["threshold_config_sha256"] = hashlib.sha256((stage / "thresholds.yaml").read_bytes()).hexdigest()
        (stage / "calibration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        stage.rename(output)
    if clean_output is not None and ready:
        try:
            if fingerprint(root) != source_identity:
                raise ValueError("Source changed before cleaning")
            if hashlib.sha256((output / "thresholds.yaml").read_bytes()).hexdigest() != report["threshold_config_sha256"]:
                raise ValueError("Generated threshold config changed before cleaning")
            # Load the published file, not an alternate in-memory threshold representation.
            cleaned = clean_v3(root, clean_output, V3Config.from_yaml(output / "thresholds.yaml"))
            report["cleaning"] = {"status": "passed", "output": str(clean_output),
                                  "report": str(clean_output / "cleaning_report/report.json"),
                                  "frames": cleaned["frames"], "episodes": cleaned["episodes"]}
        except Exception as exc:
            report["status"] = "cleaning_failed"
            report["cleaning"] = {"status": "failed", "error": str(exc)}
        (output / "calibration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
