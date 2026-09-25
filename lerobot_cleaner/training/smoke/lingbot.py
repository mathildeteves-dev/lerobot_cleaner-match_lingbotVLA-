"""Opt-in real preprocessing smoke on an already cleaned dataset; no model."""
import sys
from pathlib import Path
import yaml


def smoke_preprocessing(dataset, options, readiness, lingbot_root, norm_stats):
    if readiness.get("compatible") is not True:
        return {"status": "blocked", "error": "Training compatibility must pass first", "model_loaded": False}
    if options.entrypoint != "explicit_dataset" and readiness["contract"]["action_chunk_size"] != readiness["contract"]["model_action_chunk_size"]:
        return {"status": "blocked", "error": "Dataset/model horizons differ", "model_loaded": False}
    root = Path(lingbot_root).resolve()
    original_path = list(sys.path)
    try:
        if not (root / "lingbotvla/data/vla_data/base_dataset.py").is_file():
            raise ValueError("Missing official LingBot source")
        for name, module in tuple(sys.modules.items()):
            if name == "lingbotvla" or name.startswith("lingbotvla."):
                file = getattr(module, "__file__", None)
                if file and not Path(file).resolve().is_relative_to(root):
                    raise ValueError("A different LingBot checkout is already imported; use a fresh process")
        sys.path.insert(0, str(root))
        from lerobot_cleaner.v30.lingbot.norm import validate_norm_output
        from .batch import training_batch
        train = yaml.safe_load(options.train_config.read_text(encoding="utf-8"))
        validate_norm_output(Path(norm_stats), Path(dataset), options.robot_config, options.train_config)
        contract = readiness["contract"]
        mapping = {"mapped_dimensions": {f["target_name"]: sum(s["end"]-s["start"] for s in f["sources"])
                     for f in contract["state_features"]+contract["action_features"]},
                   "cameras": [key.removeprefix("observation.images.") for key in contract["required_cameras"]]}
        result = training_batch(Path(dataset), options.robot_config, train, Path(norm_stats), mapping, level=2)
        return {"status": "passed", **result}
    except (ImportError, OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
        return {"status": "failed", "error": str(exc), "model_loaded": False, "runtime_validated": False}
    finally:
        sys.path[:] = original_path
