"""Run LingBot's own normalization script from the correct repository directory.

Use in the Linux/WSL LingBot Python environment. --dry-run only prints the command.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml

from lerobot_cleaner.v30.lingbot.config import validate_mapping

PROJECT = Path(__file__).resolve().parents[1]


def build_command(
    lingbot_root: Path, dataset: Path, robot_config: Path, train_config: Path, norm_output: Path
) -> list[str]:
    lingbot_root, dataset = lingbot_root.resolve(), dataset.resolve()
    robot_config, train_config = robot_config.resolve(), train_config.resolve()
    for required in ["train.sh", "scripts/compute_norm.py"]:
        if not (lingbot_root / required).is_file():
            raise ValueError(f"Missing LingBot source: {lingbot_root / required}")
    validate_mapping(dataset, robot_config, train_config)
    return [
        "bash",
        "-o",
        "pipefail",
        "train.sh",
        "scripts/compute_norm.py",
        str(train_config),
        "--data.data_name",
        robot_config.stem,
        "--data.robot_config_root",
        str(robot_config.parent),
        "--data.train_path",
        str(dataset),
        "--data.norm_stats_file",
        str(norm_output.resolve()),
    ]


def single_cuda_device(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)", value):
        raise argparse.ArgumentTypeError(
            "Normalization requires exactly one GPU index or GPU/MIG UUID; GPU lists are unsupported"
        )
    return value


def validate_norm_output(path: Path, dataset: Path, robot_config: Path, train_config: Path):
    """Check statistics emitted by LingBot's full-dataset compute_norm."""
    mapping = validate_mapping(dataset, robot_config, train_config)
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    robot = yaml.safe_load(robot_config.read_text(encoding="utf-8"))
    train = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    delta_keys = {
        key for item in robot["actions"] for key, value in item.items() if value["subtract_state"]
    }
    chunk_size = train["train"].get("chunk_size", 50)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Missing or invalid normalization JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError("Normalization JSON must be an object")
    count = result.get("count")
    if type(count) is not int or count < 2 or count != info["total_frames"]:
        raise ValueError("Normalization count must equal the full dataset frame count (at least 2)")
    stats = result.get("norm_stats")
    if not isinstance(stats, dict) or set(stats) != set(mapping["mapped_dimensions"]):
        raise ValueError("Normalization feature keys do not match the robot mapping")
    for key, width in mapping["mapped_dimensions"].items():
        expected = (chunk_size, width) if key in delta_keys else (width,)
        fields = stats[key]
        if not isinstance(fields, dict):
            raise ValueError(f"Invalid normalization statistics for {key}")
        arrays = {}
        for name in ["mean", "std", "min", "max", "q01", "q02", "q98", "q99"]:
            try:
                values = np.asarray(fields.get(name))
                if (
                    values.dtype.kind not in "fiu"
                    or values.shape != expected
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Expected finite numeric values with the mapped shape")
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid normalization {key}/{name}; expected shape {expected}"
                ) from exc
            arrays[name] = values
        if (arrays["std"] < 0).any() or (arrays["min"] > arrays["max"]).any():
            raise ValueError(f"Invalid normalization standard deviation or bounds: {key}")
        for lower, upper in [("q01", "q02"), ("q02", "q98"), ("q98", "q99")]:
            if (arrays[lower] > arrays[upper]).any():
                raise ValueError(f"Invalid normalization quantile order: {key}")


def run_normalization(
    lingbot_root: Path,
    dataset: Path,
    robot_config: Path,
    train_config: Path,
    norm_output: Path,
    cuda_devices: str = "0",
):
    device = single_cuda_device(cuda_devices)
    norm = norm_output.resolve()
    if norm.exists():
        raise FileExistsError(f"Refusing to overwrite existing normalization statistics: {norm}")
    norm.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lingbot-norm-", dir=norm.parent) as folder:
        staged = Path(folder) / "norm.json"
        command = build_command(lingbot_root, dataset, robot_config, train_config, staged)
        print(shlex.join(command))
        subprocess.run(
            command,
            cwd=lingbot_root.resolve(),
            check=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": device,
                "NNODES": "1",
                "NODE_RANK": "0",
                "NPROC_PER_NODE": "1",
                "MASTER_ADDR": "127.0.0.1",
            },
        )
        validate_norm_output(staged, dataset, robot_config, train_config)
        # Publish validated data atomically without overwriting; staging is on the same filesystem.
        os.link(staged, norm)
    print(f"Validated normalization statistics: {norm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--robot-config", type=Path, default=PROJECT / "configs/robot_configs/droid_franka.yaml"
    )
    parser.add_argument(
        "--train-config", type=Path, default=PROJECT / "configs/vla/droid_franka.yaml"
    )
    parser.add_argument("--norm-output", type=Path)
    parser.add_argument(
        "--cuda-devices",
        type=single_cuda_device,
        default="0",
        help="Exactly one GPU index or GPU/MIG UUID (default: 0)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    norm = (
        args.norm_output or args.lingbot_root / f"assets/norm_stats/{args.robot_config.stem}.json"
    )
    command = build_command(
        args.lingbot_root, args.dataset, args.robot_config, args.train_config, norm
    )
    print(f"Working directory: {args.lingbot_root.resolve()}")
    if args.dry_run:
        print(
            "Single-process environment: NNODES=1 NODE_RANK=0 NPROC_PER_NODE=1 MASTER_ADDR=127.0.0.1 "
            f"CUDA_VISIBLE_DEVICES={args.cuda_devices}"
        )
        print(shlex.join(command))
        print("Actual execution stages and validates the JSON before publishing to this path.")
        return
    if os.name == "nt":
        parser.error(
            "Run this script inside the Linux/WSL LingBot environment, not Windows Python."
        )
    run_normalization(
        args.lingbot_root,
        args.dataset,
        args.robot_config,
        args.train_config,
        norm,
        args.cuda_devices,
    )


if __name__ == "__main__":
    main()
