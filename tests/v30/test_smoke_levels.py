"""Unit contracts only: these tests do not run LingBot, load models, or validate its runtime."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import smoke_lingbot as runner
from scripts.lingbot_training_batch import validate_training_batch


class TensorDouble(np.ndarray):
    def numel(self):
        return self.size


@pytest.fixture
def tensor_contract(monkeypatch):
    # NumPy-backed test double avoids importing the host's incompatible PyTorch.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=np.ndarray, isfinite=np.isfinite, bool=np.dtype(bool), int64=np.dtype("int64")))
    cfg = SimpleNamespace(max_state_dim=4, max_action_dim=4, tokenizer_max_length=6)
    batch = {
        "state": np.ones((2, 4)), "actions": np.ones((2, 3, 4)),
        "joint_mask": np.ones((2, 4), dtype=bool),
        "action_is_pad": np.zeros((2, 3), dtype=bool),
        "lang_tokens": np.ones((2, 6), dtype=np.int64),
        "lang_masks": np.ones((2, 6), dtype=bool),
        "img_masks": np.ones((2, 2), dtype=bool),
        "images": np.ones((2, 2, 3, 8, 8)),
    }
    return cfg, {key: value.view(TensorDouble) for key, value in batch.items()}


def test_training_tensor_contract(tensor_contract):
    cfg, batch = tensor_contract
    assert validate_training_batch(batch, cfg, 2, 3, 2)["actions"]["shape"] == [2, 3, 4]
    batch["images"] = np.ones((2, 2, 1, 4, 12)).view(TensorDouble)  # processor patch layout
    validate_training_batch(batch, cfg, 2, 3, 2)


@pytest.mark.parametrize("key,value", [
    ("actions", np.ones((2, 2, 4))),
    ("actions", np.full((2, 3, 4), np.nan)),
    ("joint_mask", np.zeros((2, 4), dtype=bool)),
    ("lang_tokens", np.ones((2, 6), dtype=float)),
    ("lang_tokens", -np.ones((2, 6), dtype=np.int64)),
    ("lang_masks", np.zeros((2, 6), dtype=bool)),
    ("img_masks", np.zeros((2, 2), dtype=bool)),
    ("action_is_pad", np.zeros((2, 3), dtype=float)),
    ("images", np.ones((2, 1, 3, 8, 8))),
])
def test_bad_training_batch_rejected(tensor_contract, key, value):
    cfg, batch = tensor_contract
    batch[key] = value.view(TensorDouble)
    with pytest.raises(ValueError):
        validate_training_batch(batch, cfg, 2, 3, 2)


@pytest.fixture
def smoke_contract(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/compute_norm.py").touch()
    (root / "train.sh").touch()
    robot, train, profile = [tmp_path / name for name in ("robot.yaml", "train.yaml", "profile.yaml")]
    for path in (robot, train, profile):
        path.touch()
    monkeypatch.setattr(runner, "load_profile", lambda _: None)
    monkeypatch.setattr(runner, "check_readiness", lambda *a: {"source_available": True, "missing_dependencies": []})
    monkeypatch.setattr(runner, "clean_v3", lambda *a: None)
    monkeypatch.setattr(runner, "validate_mapping", lambda *a: None)
    monkeypatch.setattr(runner, "sys", SimpleNamespace(version="3.12.3", version_info=(3, 12, 3), executable="python"))
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", environ={}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.8.0", version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True)))
    loaded = {key: True for key in ["data_loader_validated", "normalization_sample_validated",
              "getitem_validated", "dataloader_batch_validated", "training_batch_validated", "model_forward_validated"]}
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if "scripts.run_lingbot_probe" in command:
            from pathlib import Path
            Path(command[command.index("--output") + 1]).write_text(json.dumps(loaded))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    return (tmp_path / "input", tmp_path / "output", root, profile, robot, train), loaded, commands


@pytest.mark.parametrize("level", [1, 2, 3])
def test_requested_level_is_explicit_and_reported(smoke_contract, level):
    args, loaded, commands = smoke_contract
    result = runner.smoke(*args, level=level, cuda_device="2")
    assert result["status"] == "passed"
    assert result["completed_level"] == level
    assert result["model_forward_validated"] == (level == 3)
    command, kwargs = commands[-1]
    assert command[command.index("--smoke-level") + 1] == str(level)
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert ("training-shaped batch" in [s["stage"] for s in result["stages"]]) == (level >= 2)


@pytest.mark.parametrize("level,flag", [(2, "training_batch_validated"), (3, "model_forward_validated")])
def test_missing_higher_level_evidence_cannot_pass(smoke_contract, level, flag):
    args, loaded, commands = smoke_contract
    del loaded[flag]
    assert runner.smoke(*args, level=level)["status"] == "failed"


def test_invalid_level_rejected_before_work():
    with pytest.raises(ValueError, match="level"):
        runner.smoke(None, None, None, None, None, None, level=4)

@pytest.mark.parametrize("level", [2, 3])
def test_preprocessing_calls_real_interfaces_and_forward_is_opt_in(tmp_path, monkeypatch, level):
    """Interface wiring with doubles; not a real LingBot smoke result."""
    from types import ModuleType

    from scripts import lingbot_training_batch as worker
    cfg = SimpleNamespace(max_state_dim=4, max_action_dim=4, tokenizer_max_length=6)
    monkeypatch.setattr(worker, "preprocessing_config", lambda _: cfg)
    monkeypatch.setattr(worker, "validate_training_batch", lambda *a: {"validated": True})
    events = []
    utils = ModuleType("lingbotvla.data.vla_data.utils")
    utils.prepare_images = lambda item: item
    utils.prepare_language = lambda item: item
    original_images = utils.prepare_images

    class Dataset:
        def __init__(self, **kwargs):
            assert kwargs["do_nomalize"] is True
            assert kwargs["tokenizer"] == "tokenizer"
            assert kwargs["image_processor"] == "image_processor"
            assert kwargs["chunk_size"] == 3
            self.feature_transform = SimpleNamespace(
                return_item_befor_padding=False,
                normalizer=SimpleNamespace(normalize=lambda item: item),
                pad_and_concat=lambda item: item)
        def __len__(self):
            return 2
        def getdata(self, index):
            value = self.feature_transform.normalizer.normalize({"index": index})
            value = self.feature_transform.pad_and_concat(value)
            value = utils.prepare_images(value)
            return utils.prepare_language(value)
        def __getitem__(self, index):
            return self.getdata(index)

    def loader(ds, **kwargs):
        assert kwargs["num_workers"] == 0
        return [kwargs["collate_fn"]([ds[i] for i in range(kwargs["batch_size"])])]

    def processor(path, **kwargs):
        assert kwargs["local_files_only"] is True
        events.append("processor")
        return SimpleNamespace(tokenizer="tokenizer", image_processor="image_processor")

    modules = {
        "torch": {"utils": SimpleNamespace(data=SimpleNamespace(DataLoader=loader))},
        "transformers": {"AutoProcessor": SimpleNamespace(from_pretrained=processor)},
        "lingbotvla": {}, "lingbotvla.data": {},
        "lingbotvla.data.vla_data": {"utils": utils},
        "lingbotvla.data.vla_data.base_dataset": {"VLADataset": Dataset},
        "lingbotvla.data.data_transform": {"VLADataCollatorWithPacking": lambda: lambda items: items},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    from scripts import lingbot_forward

    def forward(batch, train):
        events.append("forward")
        return {"model_forward_validated": True, "model_loaded": True}
    monkeypatch.setattr(lingbot_forward, "forward_once", forward)
    train = {"model": {"tokenizer_path": "local-qwen"}, "train": {"chunk_size": 3},
             "data": {"joints": [{"arm": 4}], "cameras": ["wrist"]}}
    report = worker.training_batch(tmp_path, tmp_path / "robot.yaml", train, tmp_path / "norm.json",
                                   {"mapped_dimensions": {"action.arm": 4}}, level=level)
    assert report["preprocessing_calls"] == ["normalize", "pad_and_concat", "prepare_images", "prepare_language"]
    assert report["training_batch_validated"] is True
    assert report["model_forward_validated"] == (level == 3)
    assert events == (["processor", "forward"] if level == 3 else ["processor"])
    assert utils.prepare_images is original_images


@pytest.mark.parametrize("capacity,max_dim", [(2, 4), (8, 4), (-1, 4), (0, 4)])
def test_padding_cannot_silently_crop(capacity, max_dim):
    from scripts.lingbot_training_batch import validate_padding
    with pytest.raises(ValueError):
        validate_padding({"joints": [{"arm": capacity}]},
                         {"mapped_dimensions": {"action.arm": 4}},
                         SimpleNamespace(max_state_dim=max_dim, max_action_dim=max_dim))


def test_defaults_do_not_initialize_distributed_arguments(monkeypatch):
    from dataclasses import dataclass, field
    from types import ModuleType

    from scripts.lingbot_training_batch import preprocessing_config

    @dataclass
    class Arguments:
        max_state_dim: int = 75
        max_action_dim: int = 75
        tokenizer_max_length: int = 48
        resize_imgs_with_padding: list = field(default_factory=lambda: [224, 224])
        def __post_init__(self):
            raise AssertionError("Must not initialize training arguments")
    module = ModuleType("lingbotvla.utils.arguments")
    module.TrainingArguments = Arguments
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cfg = preprocessing_config({"tokenizer_max_length": 72})
    assert cfg.tokenizer_max_length == 72
    assert cfg.max_action_dim == 75
    assert cfg.resize_imgs_with_padding == [224, 224]

@pytest.mark.parametrize("finite", [True, False])
def test_level3_forward_once_no_grad_without_trainer(tmp_path, monkeypatch, finite):
    from contextlib import nullcontext
    from types import ModuleType

    from scripts import lingbot_forward as worker

    events = []
    class DeviceTensor(TensorDouble):
        def to(self, device):
            assert device == "cuda"
            return self
    tensor = np.ones((1, 2)).view(DeviceTensor)

    class Model:
        def eval(self):
            events.append("eval")
        def __call__(self, **kwargs):
            events.append("forward")
            assert kwargs["depth_targets"] is None
            return (tensor if finite else np.array([np.nan]).view(DeviceTensor), {"loss": tensor})

    def build(**kwargs):
        assert kwargs["init_device"] == "cuda"
        assert kwargs["weights_path"] == str(tmp_path)
        events.append("load")
        return Model()
    model_module = ModuleType("lingbotvla.models")
    model_module.build_foundation_model = build
    arguments = ModuleType("lingbotvla.utils.arguments")
    arguments.ModelArguments = type("ModelArguments", (), {})
    arguments.TrainingArguments = type("TrainingArguments", (), {})
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setitem(sys.modules, arguments.__name__, arguments)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=DeviceTensor, isfinite=np.isfinite, bfloat16="bfloat16",
        no_grad=lambda: events.append("no_grad") or nullcontext(),
        autocast=lambda *a, **k: nullcontext()))
    defaults = {"post_training": True, "enable_mixed_precision": True, "enable_fp32": False,
                "freeze_vision_encoder": False, "tokenizer_max_length": 72, "vocab_size": 0,
                "use_lm_head": False, "force_use_huggingface": False, "vlm_causal": False}
    monkeypatch.setattr(worker, "argument_defaults", lambda cls: defaults)
    (tmp_path / "test.safetensors").touch()  # file marker only; build is a test double
    train = {"model": {"model_path": str(tmp_path)}, "train": {}}
    if finite:
        result = worker.forward_once({"state": tensor}, train)
        assert result["forward_calls"] == 1
        assert result["backward_executed"] is False
        assert result["optimizer_executed"] is False
        assert result["fsdp_initialized"] is False
    else:
        with pytest.raises(ValueError, match="non-finite"):
            worker.forward_once({"state": tensor}, train)
    assert events == ["load", "eval", "no_grad", "forward"]


def test_smoke_cli_exposes_level_without_running_smoke():
    from typer.testing import CliRunner

    from lerobot_cleaner.cli import app
    result = CliRunner().invoke(app, ["smoke-lingbot", "--help"])
    assert result.exit_code == 0
    assert "level" in result.stdout

@pytest.mark.parametrize("version,compatibility", [
    ((3, 12, 3), "preferred"),
    ((3, 12, 0), "warning"),
    ((3, 12, 9), "warning"),
    ((3, 11, 9), "blocker"),
    ((3, 13, 3), "blocker"),
])
def test_python_minor_gate_and_patch_warning(smoke_contract, monkeypatch, version, compatibility):
    args, _, commands = smoke_contract
    monkeypatch.setattr(runner, "sys", SimpleNamespace(
        version=".".join(map(str, version)), version_info=version, executable="python"))
    result = runner.smoke(*args)
    assert result["environment"]["python_compatibility"] == compatibility
    assert result["environment"]["python_ok"] == (compatibility != "blocker")
    assert bool(result["warnings"]) == (compatibility == "warning")
    assert bool(result["blockers"]) == (compatibility == "blocker")
    if compatibility == "blocker":
        assert result["status"] == "blocked"
        assert not commands
        assert not args[1].exists()
    else:
        assert result["status"] == "passed"
        import json
        saved = json.loads((args[1] / "smoke_report.json").read_text(encoding="utf-8"))
        assert saved["warnings"] == result["warnings"]
