"""High-level LingBot training artifact orchestration; never implements normalization."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import yaml

from lerobot_cleaner.adapters.factory import v3_adapter
from lerobot_cleaner.adapters.lingbot_config import LingBotRobotConfigSchema
from lerobot_cleaner.training.config import TrainingCheckConfig
from lerobot_cleaner.training.compatibility.lingbot import check_training
from lerobot_cleaner.training.smoke.lingbot import smoke_preprocessing
from lerobot_cleaner.v30.v3 import V3Config, clean_v3, audit_v3
from .norm import run_normalization, validate_norm_output, single_cuda_device


def write_json(path, document):
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lingbot_provenance(root):
    required = ("train.sh", "scripts/compute_norm.py", "lingbotvla/data/vla_data/utils.py",
                "lingbotvla/data/vla_data/base_dataset.py", "lingbotvla/utils/normalize.py")
    for name in required:
        if not (root/name).is_file():
            raise ValueError(f"Missing official LingBot source: {name}")
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"root": str(root), "commit": git("rev-parse", "HEAD"),
            "worktree_status": git("status", "--porcelain"),
            "source_sha256": {name:digest(root/name) for name in required}}


def robot_document(schema):
    """Serialize existing canonical slices; the shared LingBot parser validates grammar."""
    document = {"states":[], "actions":[], "images":[]}
    for section, features, prefix in (("states", schema.states, "observation.state."),
                                       ("actions", schema.actions, "action.")):
        for feature in features:
            name = feature.name
            if section == "states" and name.startswith("state."):
                name = "observation." + name
            if not name.startswith(prefix) or name == prefix:
                raise ValueError(f"Cannot infer LingBot joint name for {name!r}; provide --robot-config or named canonical mapping")
            sources, seen = [], set()
            for i, part in enumerate(feature.slices):
                key = part.column if part.column not in seen else f"{part.column}*{i}"
                seen.add(part.column)
                sources.append({key:{"start":part.start, "end":part.end}})
            entry = {"origin_keys": sources}
            if section == "actions":
                entry.update(subtract_state=feature.subtract_state, convert_from_state=feature.convert_from_state)
            document[section].append({name:entry})
    for feature in schema.visual:
        name = "observation.images." + feature.canonical_path.removeprefix("visual.")
        if feature.source_key is not None:
            origins = feature.source_key
        else:
            origins, seen = [], set()
            for i, part in enumerate(feature.sources):
                key = part.source_key if part.source_key not in seen else f"{part.source_key}*{i}"
                seen.add(part.source_key)
                origins.append({key:{"start":part.start,"end":part.end}})
        document["images"].append({name:{"origin_keys":origins}})
    return document


def train_document(template, output, template_path):
    """Bind artifact paths, requiring explicit training choices rather than defaults."""
    if not isinstance(template, dict):
        raise ValueError("Train template must be a YAML mapping")
    train = deepcopy(template)
    for section, fields in {"model":("model_path","tokenizer_path"),
                            "data":("joints","cameras","norm_type"),
                            "train":("chunk_size","max_state_dim","max_action_dim",
                                      "tokenizer_max_length","resize_imgs_with_padding")}.items():
        if not isinstance(train.get(section), dict) or any(key not in train[section] for key in fields):
            raise ValueError(f"Train template requires explicit {section} fields: {fields}")
    # Explicit relative local assets are anchored to the template, never the norm process cwd.
    for key in ("model_path", "tokenizer_path"):
        value = train["model"][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Train template model.{key} must be nonempty")
        candidate = template_path.parent/value
        if value.startswith(("./", "../")) or candidate.exists():
            train["model"][key] = str(candidate.resolve())
    train["data"].update(data_name="robot_config", robot_config_root=str(output),
        train_path=str(output/"dataset"), norm_stats_file=str(output/"norm_stats.json"))
    return train


class LingBotBundleExporter:
    def export(self, dataset, output, *, lingbot_root, train_config, config=None,
               robot_config=None, skip_clean=False, skip_smoke=False, cuda_device="0"):
        dataset, output, root = Path(dataset).resolve(), Path(output).resolve(), Path(lingbot_root).resolve()
        train_config = Path(train_config).resolve()
        robot_config = Path(robot_config).resolve() if robot_config is not None else None
        config = config or V3Config()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite bundle: {output}")
        if output.is_relative_to(dataset) or dataset.is_relative_to(output):
            raise ValueError("Bundle output and input dataset must be disjoint")
        if not (dataset/"meta/info.json").is_file():
            raise ValueError("Missing source dataset meta/info.json")
        device = single_cuda_device(cuda_device)
        provenance = lingbot_provenance(root)
        template = yaml.safe_load(train_config.read_text(encoding="utf-8"))
        train = train_document(template, output, train_config)
        # Existing configured LingBot mapping is an explicit user choice too.
        robot_config = robot_config or config.robot_config
        if robot_config is not None:
            robot_config = Path(robot_config).resolve()
            robot_text = robot_config.read_text(encoding="utf-8")
        output.mkdir(parents=True)
        manifest = {"format":"lingbot-training-bundle-v1", "status":"building", "training_ready":False,
            "dataset":"dataset", "robot_config":"robot_config.yaml", "train_config":"train_config.yaml",
            "norm_stats":"norm_stats.json", "compatibility_report":"compatibility_report.json",
            "cleaning_report":"cleaning_report", "lingbot_version":provenance,
            "norm_backend":"official_lingbot", "norm_validated":False, "skip_clean":skip_clean, "skip_smoke":skip_smoke,
            "source_dataset":str(dataset), "train_template":str(train_config)}
        write_json(output/"manifest.json", manifest)
        readiness = {"compatible":False,"training_ready":False,"status":"not_evaluated"}
        stage = "clean"
        try:
            target = output/"dataset"
            if skip_clean:
                info = json.loads((dataset/"meta/info.json").read_text(encoding="utf-8"))
                if info.get("codebase_version") != "v3.0":
                    raise ValueError("--skip-clean requires an already prepared LeRobot v3 dataset")
                shutil.copytree(dataset, target)
            else:
                clean_v3(dataset, target, config)
            reports = output/"cleaning_report"
            if (target/"cleaning_report").is_dir():
                shutil.copytree(target/"cleaning_report", reports)
            else:
                reports.mkdir()
            stage = "validate dataset"
            quality = audit_v3(target, config, output_check=True)
            write_json(reports/"bundle_validation.json", quality)
            if quality.get("quality_policy", {}).get("abort_episodes") or quality.get("quality_policy", {}).get("rejected_episodes"):
                raise ValueError("Dataset quality policy aborted bundle validation")
            # Skip-clean never silently executes a requested rejection plan.
            if any(plan.get("reject_episode") for plan in quality.get("transform_plans", [])):
                raise ValueError("Dataset quality policy rejects episodes; clean them before exporting")
            stage = "generate configs"
            robot_path, train_path = output/"robot_config.yaml", output/"train_config.yaml"
            if robot_config is not None:
                robot_path.write_text(robot_text, encoding="utf-8")
            else:
                with v3_adapter(target, config) as adapter:
                    canonical = adapter.get_feature_schema()
                    document = robot_document(canonical)
                    LingBotRobotConfigSchema.parse(document, adapter.info["features"])
                    write_json(output/"canonical_schema.json", asdict(canonical))
                robot_path.write_text(yaml.safe_dump(document,sort_keys=False),encoding="utf-8")
            train_path.write_text(yaml.safe_dump(train,sort_keys=False,allow_unicode=True),encoding="utf-8")
            options = TrainingCheckConfig(robot_config=robot_path,train_config=train_path,
                entrypoint="official_train",decode_cameras=True, tokenization={"enabled":True})
            stage = "training compatibility"
            readiness = check_training(target, options, config)
            write_json(output/"compatibility_report.json",readiness)
            if readiness.get("compatible") is not True:
                raise ValueError("Training compatibility failed; inspect compatibility_report.json")
            stage = "official normalization"
            run_normalization(root,target,robot_path,train_path,output/"norm_stats.json",device)
            validate_norm_output(output/"norm_stats.json",target,robot_path,train_path)
            manifest["norm_validated"] = True
            stage = "Level 2 smoke"
            readiness["smoke"] = ({"status":"skipped","model_loaded":False} if skip_smoke else
                smoke_preprocessing(target,options,readiness,root,output/"norm_stats.json"))
            smoke = readiness["smoke"]
            passed = (smoke.get("status") == "passed" and smoke.get("training_batch_validated") is True
                      and smoke.get("processor_tokenizer_validated") is True and smoke.get("feature_transform_normalize") is True)
            readiness.update(runtime_validated=passed, training_ready=passed,
                             status="training_ready" if passed else "runtime_unverified")
            write_json(output/"compatibility_report.json",readiness)
            if not skip_smoke and not passed:
                raise ValueError("Level 2 smoke failed; inspect compatibility_report.json")
            # Capture the actual code used, and fail if it changed during computation.
            final_provenance = lingbot_provenance(root)
            if any(final_provenance.get(key) != provenance.get(key) for key in ("commit", "source_sha256")):
                raise ValueError("LingBot source changed while creating the bundle")
            manifest.update(status="ready" if passed else "unverified", training_ready=passed,
                artifacts_sha256={name:digest(output/name) for name in
                    ("robot_config.yaml","train_config.yaml","norm_stats.json","compatibility_report.json")})
            write_json(output/"manifest.json",manifest)
            return manifest
        except Exception as exc:
            manifest.update(status="failed",training_ready=False,failed_stage=stage,error=str(exc))
            readiness.update(training_ready=False, bundle_error={"stage":stage,"error":str(exc)})
            write_json(output/"compatibility_report.json",readiness)
            write_json(output/"manifest.json",manifest)
            raise
