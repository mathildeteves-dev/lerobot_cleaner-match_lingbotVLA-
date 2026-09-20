"""Real clean -> mapping -> compute_norm -> VLADataset -> item -> DataLoader smoke.

Run with the Python interpreter from a real Linux/WSL LingBot environment.
Levels 1/2 never load model weights. Level 3 explicitly opts into one forward.
Never downloads models, starts training, or reports mocked execution as success.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from lerobot_cleaner.v30.lingbot.config import validate_mapping
from lerobot_cleaner.v30.review_profile import load_profile
from lerobot_cleaner.v30.training_readiness import check_readiness
from lerobot_cleaner.v30.v3 import V3Config, clean_v3


def smoke(dataset, output, lingbot_root, profile_path, robot_config, train_config, cleaning_config=None, cuda_device="0", level=1):
    if level not in (1, 2, 3):
        raise ValueError("Smoke level must be 1, 2 or 3")
    paths = [dataset, output, lingbot_root, profile_path, robot_config, train_config]
    dataset, output, lingbot_root, profile_path, robot_config, train_config = map(lambda p: Path(p).resolve(), paths)
    if output == dataset or dataset in output.parents:
        raise ValueError("Smoke output must be outside the source dataset")
    if output.exists():
        raise FileExistsError(f"Smoke output already exists: {output}")
    report = {"status": "blocked", "scope": {1: "data_pipeline", 2: "training_batch", 3: "single_forward"}[level], "requested_level": level, "completed_level": 0, "model_forward_validated": False, "training_started": False, "stages": []}
    readiness = check_readiness(load_profile(profile_path), lingbot_root)
    report["preflight"] = readiness
    required = [lingbot_root / "train.sh", lingbot_root / "scripts/compute_norm.py", robot_config, train_config]
    missing = [str(path) for path in required if not path.is_file()]
    report["missing_paths"] = missing
    environment = {"python": sys.version.split()[0], "platform": os.name}
    try:
        import torch
        environment.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
    except ImportError:
        environment.update(torch=None, cuda=None, cuda_available=False)
    python_ok = sys.version_info[:2] == (3, 12)
    python_preferred = sys.version_info[:3] == (3, 12, 3)
    environment["python_ok"] = python_ok
    environment["python_compatibility"] = (
        "preferred" if python_preferred else "warning" if python_ok else "blocker"
    )
    report["warnings"] = []
    report["blockers"] = []
    if python_ok and not python_preferred:
        report["warnings"].append(
            f"Python {environment['python']} is accepted; Python 3.12.3 is preferred."
        )
    elif not python_ok:
        report["blockers"].append(
            f"Python 3.12.x is required; found {environment['python']}."
        )
    environment["matches_requested_runtime"] = (
        python_ok and os.name != "nt"
        and (environment["torch"] or "").split("+")[0] == "2.8.0"
        and environment["cuda"] == "12.8" and environment["cuda_available"]
    )
    report["environment"] = environment
    # No dataset copying/cleaning is started in an incomplete runtime.
    if missing or not readiness["source_available"] or readiness["missing_dependencies"] or not environment["matches_requested_runtime"]:
        return report
    output.mkdir(parents=True)
    clean = output / "dataset"
    norm = output / "norm.json"
    probe_report = output / "loader.json"
    try:
        config = V3Config.from_yaml(cleaning_config)
        config.verify_videos = True
        clean_v3(dataset, clean, config)
        report["stages"].append({"stage": "clean-v3", "status": "passed"})
        validate_mapping(clean, robot_config, train_config)
        report["stages"].append({"stage": "validate-lingbot", "status": "passed"})
        subprocess.run([sys.executable, "-m", "scripts.run_lingbot_norm",
            "--dataset", str(clean), "--lingbot-root", str(lingbot_root),
            "--robot-config", str(robot_config), "--train-config", str(train_config),
            "--norm-output", str(norm), "--cuda-devices", cuda_device], check=True)
        report["stages"].append({"stage": "compute_norm.py", "status": "passed"})
        from scripts.run_lingbot_norm import single_cuda_device
        probe_env = {**os.environ, "CUDA_VISIBLE_DEVICES": single_cuda_device(cuda_device),
                     "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        probe_process = subprocess.run([sys.executable, "-m", "scripts.run_lingbot_probe",
            "--dataset", str(clean), "--lingbot-root", str(lingbot_root),
            "--profile", str(profile_path), "--robot-config", str(robot_config),
            "--train-config", str(train_config), "--norm-stats", str(norm),
            "--output", str(probe_report), "--smoke-batch", "--smoke-level", str(level)], check=False, env=probe_env)
        loaded = json.loads(probe_report.read_text(encoding="utf-8"))
        report["loader_report"] = loaded
        for flag in ["data_loader_validated", "normalization_sample_validated", "getitem_validated", "dataloader_batch_validated"]:
            if loaded.get(flag) is not True:
                raise RuntimeError(f"Real loader validation incomplete: {flag}")
        report["stages"].extend({"stage": name, "status": "passed"} for name in ["VLADataset", "dataset[0]", "DataLoader batch"])
        report["completed_level"] = 1
        if level >= 2:
            if loaded.get("training_batch_validated") is not True:
                raise RuntimeError(loaded.get("error", "Level 2 training batch validation incomplete"))
            report["stages"].extend({"stage": name, "status": "passed"} for name in [
                "processor/tokenizer", "FeatureTransform(do_nomalize=True)",
                "pad_and_concat", "prepare_images", "prepare_language", "training-shaped batch"])
            report["completed_level"] = 2
        if level == 3:
            if loaded.get("model_forward_validated") is not True:
                raise RuntimeError(loaded.get("error", "Level 3 model forward validation incomplete"))
            report["stages"].append({"stage": "model forward", "status": "passed"})
            report["completed_level"] = 3
            report["model_forward_validated"] = True
        if probe_process.returncode != 0 or loaded.get("status") == "failed":
            raise RuntimeError(loaded.get("error", "Loader process failed"))
        report["status"] = "passed"
        report["loader_report"] = loaded
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        if probe_report.is_file() and "loader_report" not in report:
            try:
                report["loader_report"] = json.loads(probe_report.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report["loader_report_error"] = "Worker report is unreadable or incomplete"
    (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "output", "lingbot-root", "profile", "robot-config", "train-config"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cleaning-config", type=Path)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    report = smoke(args.dataset, args.output, args.lingbot_root, args.profile, args.robot_config, args.train_config, args.cleaning_config, args.cuda_device, level=args.level)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
