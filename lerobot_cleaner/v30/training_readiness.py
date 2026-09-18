"""Explicit blockers for controller semantics and the actual LingBot runtime."""

import importlib.util
from pathlib import Path


def check_readiness(profile, lingbot_root=None):
    blockers = []
    if not profile.semantics.verified:
        blockers.append(
            "动作模式、旋转表示、单位和夹爪约定尚未按数据来源核实；禁止推定为 DROID 关节位置。"
        )
    root = Path(lingbot_root).resolve() if lingbot_root else None
    source_ok = bool(root and (root / "lingbotvla/data/vla_data/base_dataset.py").is_file())
    if not source_ok:
        blockers.append("未提供有效 LingBot 源码目录。")
    dependencies = ["torch", "torchvision", "transformers", "datasets", "lerobot"]
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:
        blockers.append("当前 Python 缺少实际加载依赖：" + ", ".join(missing))
    blockers.append("尚未通过实际 LingBot 批次及归一化检查；运行 scripts/run_lingbot_probe.py。")
    notes = []
    if source_ok:
        eval_path = root / "experiment/libero/libero/run_libero_eval.py"
        if eval_path.exists():
            text = eval_path.read_text(encoding="utf-8")
            if "np.sign(action[..., -1])" in text:
                notes.append(
                    "本地评测代码对夹爪执行 np.sign；下载数据末维 0/1 的约定需单独核实，不能自动沿用部署转换。"
                )
    return {
        "status": "blocked",
        "runtime_validated": False,
        "source_available": source_ok,
        "lingbot_root": str(root) if root else None,
        "missing_dependencies": missing,
        "blockers": blockers,
        "source_notes": notes,
    }
