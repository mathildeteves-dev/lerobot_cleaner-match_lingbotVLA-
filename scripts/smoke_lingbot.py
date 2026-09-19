"""Real clean -> mapping -> compute_norm -> VLADataset -> item -> DataLoader smoke.

Run with the Python interpreter from a real Linux/WSL LingBot environment.
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


def smoke(dataset, output, lingbot_root, profile_path, robot_config, train_config, cleaning_config=None, cuda_device="0"):
    paths = [dataset, output, lingbot_root, profile_path, robot_config, train_config]
    dataset, output, lingbot_root, profile_path, robot_config, train_config = map(lambda p: Path(p).resolve(), paths)
    if output == dataset or dataset in output.parents:
        raise ValueError("Smoke output must be outside the source dataset")
    if output.exists():
        raise FileExistsError(f"Smoke output already exists: {output}")
    report = {"status": "blocked", "scope": "data_pipeline_smoke_only", "training_started": False, "stages": []}
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
    environment["matches_requested_runtime"] = (
        sys.version_info[:3] == (3, 12, 3) and os.name != "nt"
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
        subprocess.run([sys.executable, "-m", "scripts.run_lingbot_probe",
            "--dataset", str(clean), "--lingbot-root", str(lingbot_root),
            "--profile", str(profile_path), "--robot-config", str(robot_config),
            "--train-config", str(train_config), "--norm-stats", str(norm),
            "--output", str(probe_report), "--smoke-batch"], check=True)
        loaded = json.loads(probe_report.read_text(encoding="utf-8"))
        for flag in ["data_loader_validated", "normalization_sample_validated", "getitem_validated", "dataloader_batch_validated"]:
            if loaded.get(flag) is not True:
                raise RuntimeError(f"Real loader validation incomplete: {flag}")
        report["stages"].extend({"stage": name, "status": "passed"} for name in ["VLADataset", "dataset[0]", "DataLoader batch"])
        report["status"] = "passed"
        report["loader_report"] = loaded
    except Exception as exc:
        report.update(status="failed", error=str(exc))
    (output / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["dataset", "output", "lingbot-root", "profile", "robot-config", "train-config"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cleaning-config", type=Path)
    parser.add_argument("--cuda-device", default="0")
    args = parser.parse_args()
    report = smoke(args.dataset, args.output, args.lingbot_root, args.profile, args.robot_config, args.train_config, args.cleaning_config, args.cuda_device)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
