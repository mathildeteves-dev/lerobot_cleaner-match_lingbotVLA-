"""Opt-in Level 3: one real forward, without backward/optimizer/FSDP."""
from dataclasses import MISSING, fields
from pathlib import Path


def argument_defaults(cls):
    values = {}
    for field in fields(cls):
        if field.default is not MISSING:
            values[field.name] = field.default
        elif field.default_factory is not MISSING:
            values[field.name] = field.default_factory()
    return values


def forward_once(batch, train):
    import torch
    from lingbotvla.models import build_foundation_model
    from lingbotvla.utils.arguments import ModelArguments, TrainingArguments

    model_args = {**argument_defaults(ModelArguments), **train["model"]}
    settings = {**argument_defaults(TrainingArguments), **train["train"]}
    checkpoint = Path(model_args.get("model_path") or "")
    if not checkpoint.is_dir() or not any(checkpoint.glob("*.safetensors")):
        raise ValueError("Level 3 requires an existing local model_path with real safetensors weights")
    if model_args.get("vlm_repo_id") or not model_args["post_training"]:
        raise ValueError("Level 3 requires a full post-training checkpoint, not VLM-only initialization")
    model_args["config_path"] = model_args.get("config_path") or str(checkpoint)
    model_args["tokenizer_path"] = model_args.get("tokenizer_path") or model_args["config_path"]
    model = build_foundation_model(
        config_path=model_args["config_path"], weights_path=str(checkpoint),
        torch_dtype="float32" if settings["enable_mixed_precision"] or settings["enable_fp32"] else "bfloat16",
        init_device="cuda", freeze_vision_encoder=settings["freeze_vision_encoder"],
        tokenizer_max_length=settings["tokenizer_max_length"], vocab_size=model_args["vocab_size"],
        use_lm_head=model_args["use_lm_head"], force_use_huggingface=model_args["force_use_huggingface"],
        config_kwargs={**model_args, **settings},
    )
    model.eval()
    device_batch = {key: value.to("cuda") for key, value in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not settings["enable_fp32"]):
        outputs = model(**device_batch, vlm_causal=settings["vlm_causal"], depth_targets=None)
    count = 0

    def validate(value):
        nonlocal count
        if isinstance(value, torch.Tensor):
            if value.numel() == 0 or not torch.isfinite(value).all():
                raise ValueError("Model forward returned empty/non-finite tensor")
            count += 1
        elif isinstance(value, dict):
            for item in value.values():
                validate(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                validate(item)
    validate(outputs)
    if count == 0:
        raise ValueError("Model forward returned no tensors")
    return {"model_loaded": True, "model_forward_validated": True, "forward_calls": 1,
            "forward_output_tensors": count, "backward_executed": False,
            "optimizer_executed": False, "fsdp_initialized": False}
