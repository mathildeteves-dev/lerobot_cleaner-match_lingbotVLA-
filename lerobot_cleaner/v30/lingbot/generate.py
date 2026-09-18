"""Generate a LingBot robot config + train config pair from an embodiment spec.

The spec is a single small YAML written per robot embodiment.  Slices are
checked against the dataset's real feature dimensions at generation time, and
the staged files must pass :func:`lerobot_cleaner.v30.lingbot.config.validate_mapping`
before either file is published.  Existing files are never overwritten.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lerobot_cleaner.v30.lingbot.config import validate_mapping

# Train-section defaults copied from the verified configs/vla/droid_franka.yaml.
# They are starting points only; a spec can override every key via ``train:``.
TRAIN_DEFAULTS: dict[str, Any] = {
    "data_parallel_mode": "fsdp2",
    "enable_full_shard": False,
    "module_fsdp_enable": True,
    "use_compile": True,
    "rmpad": False,
    "rmpad_with_pos_ids": False,
    "ulysses_parallel_size": 1,
    "freeze_vision_encoder": False,
    "tokenizer_max_length": 72,
    # LingBot-vla-4b model constant; must be >= the sum of joint capacities.
    "max_action_dim": 75,
    "max_state_dim": 75,
    "lr": 5.0e-5,
    "lr_decay_style": "constant",
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "max_steps": 40000,
    "ckpt_manager": "dcp",
    "save_steps": 10000,
    "save_epochs": 0,
    "enable_fp32": True,
    "enable_resume": True,
}

# data-section keys derived from the spec; a spec must not set them by hand.
MANAGED_DATA_KEYS = {"data_name", "robot_config_root", "train_path", "joints", "cameras"}

FORBIDDEN_NAME_PREFIXES = ("observation.state.", "action.", "observation.images.")

SIDE_PREFIX = Literal["state", "action"]


class SpecSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class SpecJoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    state: list[SpecSlice] = Field(min_length=1)
    action: list[SpecSlice] = Field(min_length=1)
    # LingBot padded capacity; defaults to the mapped width (no padding).
    capacity: int | None = Field(default=None, gt=0)
    subtract_state: bool = False


class SpecCamera(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    feature: str


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: str
    tokenizer_path: str


class EmbodimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Becomes both the config filename and the LingBot data_name.
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    model: SpecModel
    joints: list[SpecJoint] = Field(min_length=1)
    cameras: list[SpecCamera] = Field(min_length=1)
    # Error when slices do not exactly tile every referenced numeric feature.
    require_full_coverage: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_unique_and_unprefixed_names(self):
        for joint in self.joints:
            if joint.name.startswith(FORBIDDEN_NAME_PREFIXES):
                raise ValueError(
                    f"Use the bare joint name without the observation.state./action. prefix: {joint.name}"
                )
        for camera in self.cameras:
            if camera.name.startswith(FORBIDDEN_NAME_PREFIXES):
                raise ValueError(f"Use the bare camera name without the observation.images. prefix: {camera.name}")
        names = [joint.name for joint in self.joints]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate joint names in spec")
        cameras = [camera.name for camera in self.cameras]
        if len(cameras) != len(set(cameras)):
            raise ValueError("Duplicate camera names in spec")
        managed = MANAGED_DATA_KEYS & set(self.data)
        if managed:
            raise ValueError(
                f"data keys are derived from the spec and cannot be overridden: {sorted(managed)}"
            )
        return self


def load_spec(path: Path) -> EmbodimentSpec:
    return EmbodimentSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _check_slices(
    spec: EmbodimentSpec, features: dict, side: SIDE_PREFIX
) -> tuple[dict[str, int], dict[str, int]]:
    """Validate every slice of one side against the real feature dimensions.

    Returns the per-joint mapped width and the per-feature mapped intervals.
    """
    widths: dict[str, int] = {}
    intervals: dict[str, list[tuple[int, int]]] = {}
    for joint in spec.joints:
        width = 0
        for item in getattr(joint, side):
            if item.feature not in features:
                raise ValueError(
                    f"Missing source feature: {item.feature} (not in dataset info.json features)"
                )
            feature = features[item.feature]
            if not feature["dtype"].startswith("float") or len(feature["shape"]) != 1:
                raise ValueError(f"Expected a floating-point vector feature: {item.feature}")
            dim = feature["shape"][0]
            if not 0 <= item.start < item.end <= dim:
                raise ValueError(f"Slice out of bounds: {item.feature}[{item.start}:{item.end}] (dim {dim})")
            width += item.end - item.start
            intervals.setdefault(item.feature, []).append((item.start, item.end))
        widths[joint.name] = width
    for feature, spans in intervals.items():
        spans = sorted(spans)
        for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
            if second_start < first_end:
                raise ValueError(f"Overlapping slices on {feature}: [{second_start}:..] inside [..:{first_end}]")
    if spec.require_full_coverage:
        for feature, spans in intervals.items():
            dim = features[feature]["shape"][0]
            cursor = 0
            for start, end in sorted(spans):
                if start > cursor:
                    raise ValueError(
                        f"Unmapped gap {feature}[{cursor}:{start}]; map every dimension or set "
                        "require_full_coverage: false"
                    )
                cursor = end
            if cursor != dim:
                raise ValueError(
                    f"Unmapped gap {feature}[{cursor}:{dim}]; map every dimension or set "
                    "require_full_coverage: false"
                )
    return widths, intervals


def _build_documents(
    spec: EmbodimentSpec,
    dataset: Path,
    state_widths: dict[str, int],
    action_widths: dict[str, int],
    robot_output: Path,
) -> tuple[dict, dict]:
    states = []
    actions = []
    capacities = []
    for joint in spec.joints:
        state_width, action_width = state_widths[joint.name], action_widths[joint.name]
        if joint.subtract_state:
            if "effector.position" in joint.name or "end.position" in joint.name:
                raise ValueError(f"LingBot forbids subtract_state for {joint.name}")
            if state_width != action_width:
                raise ValueError(
                    f"subtract_state requires equal state/action widths: {joint.name} "
                    f"({state_width} vs {action_width})"
                )
        capacity = joint.capacity if joint.capacity is not None else max(state_width, action_width)
        if max(state_width, action_width) > capacity:
            raise ValueError(
                f"Capacity {capacity} is smaller than the mapped width of {joint.name} "
                f"(state {state_width}, action {action_width})"
            )
        states.append(
            {
                f"observation.state.{joint.name}": {
                    "origin_keys": [
                        {item.feature: {"start": item.start, "end": item.end}} for item in joint.state
                    ]
                }
            }
        )
        actions.append(
            {
                f"action.{joint.name}": {
                    "origin_keys": [
                        {item.feature: {"start": item.start, "end": item.end}} for item in joint.action
                    ],
                    "subtract_state": joint.subtract_state,
                }
            }
        )
        capacities.append({joint.name: capacity})

    images = [{f"observation.images.{camera.name}": {"origin_keys": camera.feature}} for camera in spec.cameras]
    robot_doc = {"states": states, "actions": actions, "images": images}

    data_section: dict[str, Any] = {
        "datasets_type": "vla",
        "data_name": spec.name,
        "robot_config_root": str(robot_output.parent.resolve()),
        "train_path": str(Path(dataset).resolve()),
        "joints": capacities,
        "cameras": [camera.name for camera in spec.cameras],
        "num_workers": 4,
        "norm_type": "meanstd",
        "norm_stats_file": f"assets/norm_stats/{spec.name}.json",
    }
    data_section.update(spec.data)
    train_section = {"output_dir": f"output/{spec.name}", **TRAIN_DEFAULTS, **spec.train}
    train_doc = {"model": spec.model.model_dump(), "data": data_section, "train": train_section}

    total = sum(next(iter(item.values())) for item in capacities)
    for section in ["state", "action"]:
        limit = train_section[f"max_{section}_dim"]
        if total > limit:
            raise ValueError(
                f"Sum of joint capacities ({total}) exceeds train.max_{section}_dim ({limit}); "
                "raise max dim or reduce capacities"
            )
    return robot_doc, train_doc


def _dump(document: dict, header: list[str]) -> str:
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    return "".join(f"# {line}\n" for line in header) + body


def generate_configs(
    dataset: Path,
    spec: Path,
    robot_output: Path | None = None,
    train_output: Path | None = None,
) -> dict:
    """Generate and validate a LingBot config pair; never overwrite existing files."""
    dataset = Path(dataset).resolve()
    spec_path = Path(spec).resolve()
    spec = load_spec(spec_path)
    robot_output = (
        Path(robot_output).resolve()
        if robot_output is not None
        else Path.cwd() / "configs" / "robot_configs" / f"{spec.name}.yaml"
    )
    train_output = (
        Path(train_output).resolve()
        if train_output is not None
        else Path.cwd() / "configs" / "vla" / f"{spec.name}.yaml"
    )
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError("generate-lingbot-config requires LeRobot v3.0 (use clean-v3 output)")
    if robot_output == train_output:
        raise ValueError("Robot config and train config must be different files")
    if robot_output.stem != spec.name:
        raise ValueError(
            f"Robot config filename ({robot_output.stem}.yaml) must match the spec name: {spec.name}"
        )
    for output in [robot_output, train_output]:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")

    features = info["features"]
    for camera in spec.cameras:
        feature = features.get(camera.feature)
        if feature is None or feature["dtype"] != "video":
            raise ValueError(f"Missing video feature: {camera.feature}")

    state_widths, _ = _check_slices(spec, features, "state")
    action_widths, _ = _check_slices(spec, features, "action")
    robot_doc, train_doc = _build_documents(spec, dataset, state_widths, action_widths, robot_output)

    robot_header = [
        f"Generated by lerobot-cleaner generate-lingbot-config from spec: {spec_path}",
        f"Dataset: {dataset}",
        "Slices were verified against dataset features at generation time.",
        "Edit freely, then re-run validate-lingbot; switching any action to deltas",
        "(subtract_state) also requires updating deployment and recomputing norm stats.",
    ]
    train_header = [
        f"Generated by lerobot-cleaner generate-lingbot-config from spec: {spec_path}",
        "Hyperparameters are unverified starting points; adjust for your hardware.",
        f"Next step: scripts/run_lingbot_norm.py --dataset {dataset}",
    ]

    robot_output.parent.mkdir(parents=True, exist_ok=True)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    # Stage under the final robot name so validate_mapping's data_name == filename
    # check holds; train gets a distinct name because both normally share a stem
    # and would otherwise overwrite each other in the same temp folder.
    with tempfile.TemporaryDirectory(prefix=".lingbot-generate-", dir=robot_output.parent) as folder:
        staged_robot = Path(folder) / robot_output.name
        staged_train = Path(folder) / f"train-{train_output.name}"
        staged_robot.write_text(_dump(robot_doc, robot_header), encoding="utf-8")
        staged_train.write_text(_dump(train_doc, train_header), encoding="utf-8")
        # Re-check destinations: nothing is allowed to appear between the entry check and publish.
        for output in [robot_output, train_output]:
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {output}")
        result = validate_mapping(dataset, staged_robot, staged_train)
        shutil.move(str(staged_robot), str(robot_output))
        shutil.move(str(staged_train), str(train_output))

    result.update(
        {
            "robot_config": str(robot_output),
            "train_config": str(train_output),
            "next_step": "Run scripts/run_lingbot_norm.py to compute normalization statistics.",
        }
    )
    return result
