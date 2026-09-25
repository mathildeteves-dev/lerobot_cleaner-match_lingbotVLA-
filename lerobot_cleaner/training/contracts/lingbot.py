"""Pinned LingBot feature + sampling contract, without importing model code."""
import ast
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import numpy as np
import yaml
from lerobot_cleaner.adapters.lingbot_config import LingBotRobotConfigSchema

# Audited TrainingArguments / VLADataset defaults in Robbyant commit 4eb34b7.
DEFAULTS = {"chunk_size": 50, "max_state_dim": 32, "max_action_dim": 32,
            "tokenizer_max_length": 48, "resize_imgs_with_padding": [224, 224]}
REVISION = "4eb34b7693a0565c67433f8fac9c59a2e67eb60b"


def finding(code, message, severity="ERROR", **details):
    return {"code": code, "severity": severity, "message": message, **details}


@dataclass
class LingBotTrainingContract:
    fps: float
    action_chunk_size: int
    model_action_chunk_size: int
    state_padded_dim: int
    action_padded_dim: int
    joints: dict
    required_cameras: list
    schema: LingBotRobotConfigSchema
    raw_features: dict
    tokenizer_max_length: int
    image_size: list
    entrypoint: str
    provenance: dict
    findings: list = field(default_factory=list)
    required_task_text: bool = True

    @classmethod
    def build(cls, info, schema, training, entrypoint="official_train"):
        settings = {**DEFAULTS, **training.get("train", {})}
        for key in ("chunk_size", "max_state_dim", "max_action_dim", "tokenizer_max_length"):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f"Training contract: {key} must be a positive integer")
        image_size = settings["resize_imgs_with_padding"]
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2 or any(type(v) is not int or v < 1 for v in image_size):
            raise ValueError("Training contract: resize_imgs_with_padding requires two positive dimensions")
        if entrypoint not in {"official_train", "explicit_dataset"}:
            raise ValueError("Training contract: unsupported entrypoint")
        data = training.get("data", {})
        joints = {}
        for entry in data.get("joints", []):
            entry = ast.literal_eval(entry) if isinstance(entry, str) else entry
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError("Training contract: joint entries require one name/capacity")
            name, capacity = next(iter(entry.items()))
            if not isinstance(name, str) or type(capacity) is not int or capacity < 0 or name in joints:
                raise ValueError("Training contract: invalid/duplicate joint capacity")
            if capacity:
                joints[name] = capacity
        cameras = data.get("cameras", [])
        if not isinstance(cameras, list) or any(not isinstance(c, str) for c in cameras) or len(set(cameras)) != len(cameras):
            raise ValueError("Training contract: cameras require unique names")
        fps = float(info["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Training contract: fps must be positive and finite")
        result = cls(fps, DEFAULTS["chunk_size"] if entrypoint == "official_train" else settings["chunk_size"],
            settings["chunk_size"], settings["max_state_dim"], settings["max_action_dim"], joints,
            ["observation.images." + c for c in cameras], schema, info["features"],
            settings["tokenizer_max_length"], settings["resize_imgs_with_padding"], entrypoint,
            {"lingbot_revision": REVISION, "lerobot_version": "0.4.2", "fps": "official dataset metadata",
             "parameters": {k: "train config" if k in training.get("train", {}) else "audited LingBot default" for k in DEFAULTS},
             "dataset_chunk": "VLADataset default (official train does not pass chunk_size)" if entrypoint == "official_train" else "explicit constructor argument"})
        result.provenance["tokenizer_path"] = training.get("model", {}).get("tokenizer_path")
        if result.action_chunk_size != result.model_action_chunk_size:
            result.findings.append(finding("horizon_mismatch", "Official training entry does not forward train.chunk_size to VLADataset"))
        if not joints or not schema.actions:
            result.findings.append(finding("no_actions", "At least one active action joint is required"))
        for section, prefix in ((schema.states, "observation.state."), (schema.actions, "action.")):
            for feature in section:
                name = feature.target_name.removeprefix(prefix)
                if feature.target_name != prefix + name or name not in joints:
                    result.findings.append(finding("unconsumed_target", f"{feature.target_name} is not consumed by training joints"))
                elif feature.dimension > joints[name]:
                    result.findings.append(finding("joint_capacity_exceeded", f"{feature.target_name} would be truncated by negative F.pad"))
        total = sum(joints.values())
        if total > result.state_padded_dim or total > result.action_padded_dim:
            result.findings.append(finding("model_capacity_exceeded", "Joint layout exceeds model padded dimensions; official F.pad would crop"))
        shared = result.source_keys("states") & result.source_keys("actions")
        if shared:
            result.findings.append(finding("state_action_source_rank_collision", "Action delta queries also expand state sources to H x D", sources=sorted(shared)))
        for image in schema.images:
            if image.target_name not in result.required_cameras:
                result.findings.append(finding("unconsumed_camera", f"{image.target_name} is absent from training cameras"))
        if not cameras:
            result.findings.append(finding("no_cameras", "Official image preparation needs at least one camera"))
        if settings.get("align_params"):
            result.findings.append(finding("depth_runtime_required", "Depth alignment requires its own runtime validation", "WARNING"))
        if data.get("dataloader_type", "native") != "native" or data.get("datasets_type", "vla") != "vla":
            result.findings.append(finding("unsupported_data_path", "Contract covers native VLA training only"))
        if settings.get("rmpad"):
            result.findings.append(finding("unsupported_rmpad", "Official VLA entry rejects rmpad"))
        return result

    def source_keys(self, section):
        return {s.origin_key for f in getattr(self.schema, section) for s in f.sources}

    def delta_timestamps(self):
        return {key: [i / self.fps for i in range(self.action_chunk_size)] for key in sorted(self.source_keys("actions"))}

    def layout(self, section):
        prefix = "observation.state." if section == "states" else "action."
        features = {f.target_name: f for f in getattr(self.schema, section)}
        padded = self.state_padded_dim if section == "states" else self.action_padded_dim
        mask, offsets = [], []
        for name, capacity in self.joints.items():
            feature = features.get(prefix + name)
            dim = feature.dimension if feature else 0
            offsets.append({"joint": name, "start": len(mask), "capacity": capacity, "mapped_dim": dim})
            mask.extend([True] * min(dim, capacity) + [False] * max(0, capacity - dim))
        compatible = len(mask) <= padded and all(r["mapped_dim"] <= r["capacity"] for r in offsets)
        mask += [False] * max(0, padded - len(mask))
        sources = self.source_keys(section)
        return {"raw_dim": sum(int(np.prod(self.raw_features[k]["shape"])) for k in sources),
                "mapped_dim": sum(f.dimension for f in features.values()), "joint_layout_dim": sum(self.joints.values()),
                "padded_dim": padded, "expected_dim": padded, "compatible": compatible,
                "layout": offsets, "dimension_mask": mask}

    def to_dict(self):
        return {"target": "lingbot_vla", "fps": self.fps, "action_chunk_size": self.action_chunk_size,
            "model_action_chunk_size": self.model_action_chunk_size, "action_horizon": self.action_chunk_size,
            "last_offset_seconds": (self.action_chunk_size-1)/self.fps,
            "delta_timestamps": self.delta_timestamps(), "required_task_text": True,
            "state_features": [asdict(f) for f in self.schema.states],
            "action_features": [asdict(f) for f in self.schema.actions],
            "camera_features": [asdict(f) for f in self.schema.images],
            "required_cameras": self.required_cameras, "state": self.layout("states"), "action": self.layout("actions"),
            "joint_mask": {"shape": [self.action_padded_dim], "dtype": "bool", "true_means": "real action dimension",
                           "values": self.layout("actions")["dimension_mask"]},
            "action_is_pad": {"shape": [self.action_chunk_size], "dtype": "bool", "true_means": "requested index outside episode"},
            "loss_mask_note": "Audited LingBot forward uses joint_mask but does not apply action_is_pad to loss; padding statistics do not imply ignored loss slots",
            "padding": "indices clamped to episode endpoints; dimensions zero padded in training joint order",
            "image_requirements": {"layout": "CHW", "loader_resize": [224, 224], "model_resize": self.image_size},
            "language": {"source": "meta.tasks.iloc[task_index].name", "bos_and_newline": True,
                         "tokenizer_max_length": self.tokenizer_max_length,
                         "padding": "max_length", "padding_side": "right", "truncation": True}, "provenance": self.provenance,
            "normalization": "requires matching norm_stats_file; validated by runtime smoke"}


def load_contract(info, options):
    # Official VLADataset resizes each source image BEFORE FeatureTransform.
    mapping_features = deepcopy(info["features"])
    for spec in mapping_features.values():
        if spec.get("dtype") in {"image", "video"} and len(spec.get("shape", [])) == 3:
            spec["shape"] = [224, 224, spec["shape"][-1]]
    schema = LingBotRobotConfigSchema.from_yaml(options.robot_config, mapping_features, allow_missing_images=True)
    training = yaml.safe_load(options.train_config.read_text(encoding="utf-8"))
    if not isinstance(training, dict):
        raise ValueError("Training config must be a mapping")
    contract = LingBotTrainingContract.build(info, schema, training, options.entrypoint)
    if training.get("data", {}).get("data_name") != options.robot_config.stem:
        contract.findings.append(finding("data_name_mismatch", "Training data_name must select the supplied robot YAML"))
    contract.provenance.update(robot_config=str(options.robot_config), train_config=str(options.train_config))
    return contract
