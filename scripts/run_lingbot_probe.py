"""Read actual LingBot samples, without its random error-retry fallback.

Run in the LingBot environment. Uses explicit robot/train configs. Missing
dependencies, source, semantic evidence or normalization are never reported as ready.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from lerobot_cleaner.v30.lingbot.config import validate_mapping
from lerobot_cleaner.v30.review_profile import load_profile
from lerobot_cleaner.v30.training_readiness import check_readiness


def probe(dataset, lingbot_root, profile, robot_config, train_config, norm_stats=None, smoke_batch=False):
    result = check_readiness(profile, lingbot_root)
    if not result["source_available"] or result["missing_dependencies"]:
        return result
    mapping = validate_mapping(dataset, robot_config, train_config)
    train = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(lingbot_root.resolve()))
    import torch
    from lingbotvla.data.vla_data.base_dataset import VLADataset

    data = dict(train["data"])
    data["joints"] = [str(item) if isinstance(item, dict) else item for item in data["joints"]]
    data["norm_stats_file"] = str(norm_stats.resolve()) if norm_stats else None
    chunk_size = train["train"].get("chunk_size", 50)
    ds = VLADataset(
        repo_id=str(dataset.resolve()),
        data_name=robot_config.stem,
        data_config=SimpleNamespace(**data),
        robot_config_root=str(robot_config.parent.resolve()),
        config=None,
        do_nomalize=False,
        chunk_size=chunk_size,
    )
    # Actual underlying LeRobot image queries and LingBot mapping, not a substitute reader.
    ds.dataset.load_image = True
    ds.feature_transform.load_image = True
    if norm_stats:
        from scripts.run_lingbot_norm import validate_norm_output

        validate_norm_output(norm_stats, dataset, robot_config, train_config)
        ds.feature_transform.normalizer = ds.feature_transform.get_normalizer(
            str(norm_stats), True, ds.feature_transform.data_config
        )
    sampled = []
    for index in sorted({0, len(ds) // 2, len(ds) - 1}):
        item = ds.getdata(index)  # Avoid __getitem__ silently substituting another trajectory.
        if not isinstance(item.get("task"), str) or not item["task"].strip():
            raise ValueError("Actual loader returned blank language")
        for key, width in mapping["mapped_dimensions"].items():
            values = np.asarray(item[key])
            expected = (chunk_size, width) if key.startswith("action.") else (width,)
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(
                    f"Actual loader feature mismatch: {key}, {values.shape}, expected {expected}"
                )
        for camera in mapping["cameras"]:
            values = np.asarray(item["observation.images." + camera])
            if values.ndim != 3 or values.shape[0] != 3 or not np.isfinite(values).all():
                raise ValueError(f"Actual loader image mismatch: {camera}")
        # Exercise torch collation on the actual transformed item.
        batch = torch.utils.data.default_collate([item])
        sampled.append(
            {
                "index": index,
                "task": item["task"],
                "tensor_shapes": {
                    k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")
                },
            }
        )
    batch_shapes = None
    if smoke_batch:
        if not norm_stats:
            raise ValueError("Smoke batch requires validated normalization statistics")
        # Audit every getdata request made by __getitem__; retries/substitutions fail smoke.
        requested = []
        original_getdata = ds.getdata

        def traced_getdata(index):
            requested.append(int(index))
            return original_getdata(index)

        ds.getdata = traced_getdata
        ds[0]
        if requested != [0]:
            raise ValueError(f"dataset[0] retried or substituted indices: {requested}")
        requested.clear()
        size = min(2, len(ds))
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=size, shuffle=False, num_workers=0)))
        if requested != list(range(size)):
            raise ValueError(f"DataLoader retried or substituted indices: {requested}")
        for key, width in mapping["mapped_dimensions"].items():
            values = batch[key]
            expected = (size, chunk_size, width) if key.startswith("action.") else (size, width)
            if tuple(values.shape) != expected or not torch.isfinite(values).all():
                raise ValueError(f"Invalid DataLoader tensor {key}: {values.shape}")
        batch_shapes = {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}

    blockers = []
    if not profile.semantics.verified:
        blockers.append("实际字段加载已通过，但控制语义尚未按数据来源核实。")
    if not norm_stats:
        blockers.append("未提供经过全数据验证的 LingBot 归一化统计。")
    blockers.append(
        "尚未验证模型分词、图像处理及训练前向；本命令仅验证实际数据加载、字段映射、可选归一化与批次拼接。"
    )
    return {
        **result,
        "status": "blocked",
        "data_loader_validated": True,
        "getitem_validated": smoke_batch,
        "dataloader_batch_validated": smoke_batch,
        "batch_shapes": batch_shapes,
        "normalization_sample_validated": bool(norm_stats),
        "runtime_validated": False,
        "blockers": blockers,
        "samples": sampled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--smoke-batch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output report already exists; select a new path")
    profile = load_profile(args.profile)
    try:
        result = check_readiness(profile, args.lingbot_root)
        if not args.robot_config or not args.train_config:
            result["blockers"].append(
                "实际加载需要显式 --robot-config 和 --train-config，避免误用 DROID 映射。"
            )
        else:
            result = probe(
                args.dataset,
                args.lingbot_root,
                profile,
                args.robot_config,
                args.train_config,
                args.norm_stats,
                smoke_batch=args.smoke_batch,
            )
    except Exception as exc:
        result = {"status": "failed", "runtime_validated": False, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("data_loader_validated") else 2


if __name__ == "__main__":
    raise SystemExit(main())
