"""Level 2: real LingBot preprocessing, without constructing a model or trainer."""
import ast
from contextlib import ExitStack
from dataclasses import MISSING, fields
from types import SimpleNamespace
from unittest.mock import patch


def preprocessing_config(train):
    # Read official defaults without TrainingArguments.__post_init__ (distributed env).
    from lingbotvla.utils.arguments import TrainingArguments

    names = {"max_state_dim", "max_action_dim", "resize_imgs_with_padding", "tokenizer_max_length"}
    values = {}
    for field in fields(TrainingArguments):
        if field.name in names:
            values[field.name] = (field.default if field.default is not MISSING
                                  else field.default_factory())
    values.update({key: train[key] for key in names if key in train})
    if set(values) != names:
        raise ValueError("Unsupported LingBot preprocessing configuration")
    return SimpleNamespace(**values)


def validate_training_batch(batch, config, size, chunk_size, cameras):
    import torch

    expected = {
        "state": (size, config.max_state_dim),
        "actions": (size, chunk_size, config.max_action_dim),
        "joint_mask": (size, config.max_action_dim),
        "action_is_pad": (size, chunk_size),
        "lang_tokens": (size, config.tokenizer_max_length),
        "lang_masks": (size, config.tokenizer_max_length),
        "img_masks": (size, cameras),
    }
    for key, shape in expected.items():
        value = batch.get(key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Training batch shape mismatch: {key}, expected {shape}")
    images = batch.get("images")
    # Qwen processors produce patch tensors; non-Qwen processors produce CHW.
    if not isinstance(images, torch.Tensor) or images.ndim < 4 or tuple(images.shape[:2]) != (size, cameras):
        raise ValueError("Training batch image/camera dimensions mismatch")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and (value.numel() == 0 or not torch.isfinite(value).all()):
            raise ValueError(f"Empty or non-finite training tensor: {key}")
    for key in ("joint_mask", "action_is_pad", "lang_masks", "img_masks"):
        if batch[key].dtype != torch.bool:
            raise ValueError(f"Training mask must be boolean: {key}")
    if batch["lang_tokens"].dtype != torch.int64 or (batch["lang_tokens"] < 0).any():
        raise ValueError("Invalid language token IDs")
    if not batch["lang_masks"].any(1).all() or not batch["joint_mask"].any(1).all():
        raise ValueError("Empty language or joint mask")
    if not batch["img_masks"].all():
        raise ValueError("Configured camera missing from training batch")
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in batch.items() if isinstance(value, torch.Tensor)}


def validate_padding(data, mapping, config):
    total = 0
    for entry in data["joints"]:
        joint = ast.literal_eval(entry) if isinstance(entry, str) else entry
        if not isinstance(joint, dict) or len(joint) != 1:
            raise ValueError("Each configured joint must have one padding capacity")
        name, capacity = next(iter(joint.items()))
        if type(capacity) is not int or capacity < 0:
            raise ValueError(f"Invalid joint padding capacity: {name}")
        if capacity == 0:
            continue  # Official FeatureInfo excludes disabled groups.
        for prefix in ("observation.state.", "action."):
            if mapping["mapped_dimensions"].get(prefix + name, 0) > capacity:
                raise ValueError(f"Joint padding would truncate {prefix + name}")
        total += capacity
    if not 0 < total <= min(config.max_state_dim, config.max_action_dim):
        raise ValueError("Combined joint padding exceeds model dimensions or has no active joints")


def training_batch(dataset, robot_config, train, norm_stats, mapping, *, level=2):
    if level not in (2, 3):
        raise ValueError("Training batch requires level 2 or 3")
    import torch
    from lingbotvla.data.data_transform import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data import utils
    from lingbotvla.data.vla_data.base_dataset import VLADataset
    from transformers import AutoProcessor

    # Caller validates the full statistics against the original train config first.
    settings = train["train"]
    if settings.get("align_params"):
        raise ValueError("Level 2 currently supports RGB VLA only; depth alignment needs a separate smoke")
    if settings.get("rmpad"):
        raise ValueError("Official LingBot VLA does not support rmpad")
    config = preprocessing_config(settings)
    tokenizer_path = train["model"]["tokenizer_path"]
    processor = AutoProcessor.from_pretrained(
        tokenizer_path, padding_side="right", trust_remote_code=True, local_files_only=True,
    )
    data = dict(train["data"])
    data["joints"] = [str(item) if isinstance(item, dict) else item for item in data["joints"]]
    data["norm_stats_file"] = str(norm_stats.resolve())
    validate_padding(data, mapping, config)
    chunk_size = settings.get("chunk_size", 50)
    ds = VLADataset(
        repo_id=str(dataset.resolve()), data_name=robot_config.stem,
        robot_config_root=str(robot_config.parent.resolve()), data_config=SimpleNamespace(**data),
        config=config, tokenizer=processor.tokenizer,
        image_processor=processor.image_processor if "qwen" in tokenizer_path.lower() else None,
        do_nomalize=True, chunk_size=chunk_size, use_depth_align=False,
    )
    transform = ds.feature_transform
    if transform.return_item_befor_padding or transform.normalizer is None:
        raise ValueError("FeatureTransform normalization/training path is inactive")
    # Observe actual calls; do not replace any preprocessing implementation.
    calls = []

    def traced(name, function):
        def invoke(*args, **kwargs):
            value = function(*args, **kwargs)
            calls.append(name)
            return value
        return invoke

    requested = []
    original = ds.getdata

    def getdata(index):
        requested.append(int(index))
        if requested != list(range(len(requested))):
            raise ValueError(f"Loader retried/substituted samples: {requested}")
        return original(index)

    ds.getdata = getdata
    with ExitStack() as stack:
        for owner, name in [(transform.normalizer, "normalize"), (transform, "pad_and_concat"),
                            (utils, "prepare_images"), (utils, "prepare_language")]:
            stack.enter_context(patch.object(owner, name, traced(name, getattr(owner, name))))
        # Direct getdata fails immediately rather than entering the loader's retry loop.
        item = ds.getdata(0)
        sequence = ["normalize", "pad_and_concat", "prepare_images", "prepare_language"]
        if calls != sequence:
            raise ValueError(f"Unexpected FeatureTransform path: {calls}")
        collator = VLADataCollatorWithPacking()
        validate_training_batch(collator([item]), config, 1, chunk_size, len(data["cameras"]))
        requested.clear()
        calls.clear()
        size = min(2, len(ds))
        batch = next(iter(torch.utils.data.DataLoader(
            ds, batch_size=size, shuffle=False, num_workers=0, collate_fn=collator,
        )))
        if requested != list(range(size)) or calls != sequence * size:
            raise ValueError("Training DataLoader did not follow the validated preprocessing path")
    tensors = validate_training_batch(batch, config, size, chunk_size, len(data["cameras"]))
    result = {
        "training_batch_validated": True, "model_forward_validated": False,
        "processor_tokenizer_validated": True, "feature_transform_normalize": True,
        "preprocessing_calls": sequence, "training_batch_tensors": tensors,
        "training_batch_size": size, "model_loaded": False,
        "backward_executed": False, "optimizer_executed": False, "fsdp_initialized": False,
    }
    if level == 3:
        from scripts.lingbot_forward import forward_once
        try:
            result.update(forward_once(batch, train))
        except Exception as exc:
            result.update(error=str(exc), failed_stage="model forward", model_loaded=None)
    return result
