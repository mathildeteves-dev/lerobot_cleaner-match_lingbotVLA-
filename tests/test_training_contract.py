"""Independent training diagnostics plus real upstream method conformance."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from lerobot_cleaner.adapters.lingbot_config import LingBotRobotConfigSchema
from lerobot_cleaner.training.config import TrainingCheckConfig
from lerobot_cleaner.training.contracts.lingbot import LingBotTrainingContract, load_contract
from lerobot_cleaner.training.compatibility.chunks import sample_indices, chunk_statistics
from lerobot_cleaner.training.compatibility.lingbot import evaluate_storage, task_text_statistics
from test_lingbot_conformance import upstream

PROJECT = Path(__file__).resolve().parents[1]


def options(tmp_path, **kwargs):
    return TrainingCheckConfig(robot_config=tmp_path / "robot.yaml", train_config=tmp_path / "train.yaml", **kwargs)


def inputs(state_dim=7, action_dim=7, expected=32, capacity=None, horizon=50):
    features = {"q": {"dtype": "float32", "shape": [state_dim]},
                "u": {"dtype": "float32", "shape": [action_dim]},
                "cam": {"dtype": "image", "shape": [4, 5, 3]}}
    robot = {"states": [{"observation.state.arm": {"origin_keys": "q"}}],
             "actions": [{"action.arm": {"origin_keys": "u"}}],
             "images": [{"observation.images.top": {"origin_keys": "cam"}}]}
    train = {"data": {"data_name": "robot", "joints": [{"arm": capacity or max(state_dim, action_dim)}], "cameras": ["top"]},
             "train": {"max_state_dim": expected, "max_action_dim": expected, "chunk_size": horizon}}
    return {"fps": 10, "features": features}, robot, train


def contract(**kwargs):
    info, robot, train = inputs(**kwargs)
    return LingBotTrainingContract.build(info, LingBotRobotConfigSchema.parse(robot, info["features"]), train)


class Storage:
    def __init__(self, root, info, lengths=(20,)):
        self.root = root
        self.info = {**info, "total_episodes": len(lengths)}
        self.tasks = pd.DataFrame({"task_index": [0]}, index=["pick the block"])
        self.frames = []
        for i, length in enumerate(lengths):
            self.frames.append(pd.DataFrame({"episode_index": i, "timestamp": np.arange(length)/info["fps"],
                "task_index": np.zeros(length, dtype=np.int64), "q": [np.zeros(info["features"]["q"]["shape"])]*length,
                "u": [np.zeros(info["features"]["u"]["shape"])]*length,
                "cam": [np.zeros((4, 5, 3), dtype=np.uint8)]*length}))
    def episode(self, i):
        return self.frames[i], {"episode_index": i, "length": len(self.frames[i])}
    def path(self, value):
        return self.root / value


@pytest.mark.parametrize("length,horizon,expected", [(10000,50,.00245),(60,50,1225/3000),(20,50,.79),(1,50,.98),(20,1,0)])
def test_A_B_C_exact_boundary_padding(length,horizon,expected):
    result = chunk_statistics(length,horizon,True)
    assert result["action_chunk_padding_ratio"] == pytest.approx(expected)
    assert result["fully_padded_samples"] == 0
    assert result["usable_action_steps"] + result["padded_action_steps"] == length*horizon
    assert result["padding_by_timestep"][-1] == (horizon-1)/horizon


def test_short_episode_only_warns_and_never_changes_quality_or_rows(tmp_path):
    c = contract()
    storage = Storage(tmp_path, {"fps": c.fps, "features": c.raw_features}, (20,60))
    before = [frame.copy(deep=True) for frame in storage.frames]
    report = evaluate_storage(storage,c,options(tmp_path,padding_warning_ratio=.5))
    assert report["compatible"] and report["status"] == "static_compatible"
    assert report["episodes_excluded"] == [] and not report["mutation_performed"]
    assert any(f["code"] == "high_boundary_padding" and f["severity"] == "WARNING" for f in report["findings"])
    assert report["action_chunk"]["total_action_slots"] == 4000
    assert report["action_chunk"]["total_padded_action_slots"] == 2015
    assert "dataset_quality" not in report
    for left,right in zip(before,storage.frames):
        pd.testing.assert_frame_equal(left,right)


@pytest.mark.parametrize("text,code", [(None,"null_task_text"),("", "empty_task_text"),("  ","empty_task_text"),(42,"wrong_type_task_text")])
def test_D_task_contract_uses_official_index_text(tmp_path,text,code):
    c=contract(); storage=Storage(tmp_path,{"fps":10,"features":c.raw_features})
    storage.tasks=pd.DataFrame({"task_index":[0],"task":["unused fallback"]},index=[text])
    report=evaluate_storage(storage,c,options(tmp_path))
    assert not report["compatible"] and report["task_text"]["counts"][code] == 20


@pytest.mark.parametrize("indices", [[None],[2],[0.5]])
def test_missing_task_mapping(indices):
    report,_=task_text_statistics(pd.DataFrame({"task_index":indices}),pd.DataFrame({"task_index":[0]},index=["pick"]))
    assert report["task_text_missing_ratio"] == 1


def test_E_missing_required_camera_is_training_error(tmp_path):
    info,robot,train=inputs(); train["data"]["cameras"].append("wrist")
    c=LingBotTrainingContract.build(info,LingBotRobotConfigSchema.parse(robot,info["features"]),train)
    report=evaluate_storage(Storage(tmp_path,info),c,options(tmp_path))
    assert report["cameras"]["missing"] == ["observation.images.wrist"]
    assert not report["compatible"] and report["episodes_excluded"] == []


@pytest.mark.parametrize("state,action,expected,ok", [(7,7,32,True),(32,32,32,True),(40,7,32,False),(7,40,32,False)])
def test_F_G_H_dimensions(tmp_path,state,action,expected,ok):
    c=contract(state_dim=state,action_dim=action,expected=expected)
    report=evaluate_storage(Storage(tmp_path,{"fps":10,"features":c.raw_features}),c,options(tmp_path))
    assert report["compatible"] == ok
    assert report["state"]["mapped_dim"] == state
    assert report["action"]["mapped_dim"] == action


def test_I_joint_mask_action_semantics(tmp_path):
    c=contract(state_dim=3,action_dim=7,capacity=8)
    assert c.layout("states")["dimension_mask"] == [True]*3+[False]*29
    assert c.to_dict()["joint_mask"]["values"] == [True]*7+[False]*25


def test_J_camera_temporal_gap(tmp_path):
    c=contract(); storage=Storage(tmp_path,{"fps":10,"features":c.raw_features})
    storage.frames[0].at[5,"cam"]=None
    report=evaluate_storage(storage,c,options(tmp_path))
    assert report["cameras"]["camera_availability_ratio"] == .95
    assert not report["compatible"]


def test_K_mapped_action_dimension_and_state_chunk_rank_collision(tmp_path):
    info,robot,train=inputs()
    robot["actions"][0]["action.arm"]={"origin_keys":[{"q":{"start":2,"end":4}},{"u":{"start":0,"end":1}}],"convert_from_state":True}
    c=LingBotTrainingContract.build(info,LingBotRobotConfigSchema.parse(robot,info["features"]),train)
    assert c.layout("actions")["mapped_dim"] == 3
    assert c.source_keys("actions") == {"q","u"}
    assert any(f["code"] == "state_action_source_rank_collision" for f in c.findings)


def test_official_entrypoint_does_not_forward_configured_horizon():
    c=contract(horizon=20)
    assert c.action_chunk_size == 50 and c.model_action_chunk_size == 20
    assert any(f["code"] == "horizon_mismatch" for f in c.findings)


def test_contract_load_reuses_grammar_and_diagnoses_missing_source(tmp_path):
    info,robot,train=inputs(); robot["images"].append("observation.images.missing")
    train["data"]["cameras"].append("missing")
    opts=options(tmp_path)
    opts.robot_config.write_text(yaml.safe_dump(robot),encoding="utf-8")
    opts.train_config.write_text(yaml.safe_dump(train),encoding="utf-8")
    c=load_contract(info,opts)
    assert not evaluate_storage(Storage(tmp_path,info),c,opts)["compatible"]
    with pytest.raises(ValueError):
        LingBotRobotConfigSchema.from_yaml(opts.robot_config,info["features"])


def source_method(path,class_name,method_name,namespace):
    """Compile the unmodified upstream method AST; no dependency-heavy package init."""
    if not path.is_file():
        pytest.skip(f"Official source required: {path}")
    tree=ast.parse(path.read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name) if class_name else tree
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),namespace)
    return namespace[method_name]


@pytest.mark.parametrize("length",[1,20,60,1000])
@pytest.mark.parametrize("fps",[10,29.97])
def test_official_sampling_indices_chunks_and_masks(length,fps):
    import torch
    root=Path(os.environ.get("LINGBOT_ROOT",PROJECT.parent.parent/"lingbot-vla"))
    delta=source_method(root/"lingbotvla/data/vla_data/base_dataset.py","VLADataset","get_delta_timestamps",{})
    c=contract(); c.fps=fps
    timestamps=delta(SimpleNamespace(feature_transform=SimpleNamespace(org_features={"actions":["u"]}),dataset_meta=SimpleNamespace(fps=fps),chunk_size=50))
    assert timestamps==c.delta_timestamps()
    path=Path(os.environ.get("LEROBOT_DATASET_SOURCE",PROJECT/"outputs/official-reference/lerobot_dataset.py"))
    query=source_method(path,"LeRobotDataset","_get_query_indices",{"torch":torch})
    offset=120
    utils_path=Path(os.environ.get("LEROBOT_UTILS_SOURCE",PROJECT/"outputs/official-reference/lerobot_utils.py"))
    get_indices=source_method(utils_path,None,"get_delta_indices",{})
    obj=SimpleNamespace(meta=SimpleNamespace(episodes=[{"dataset_from_index":offset,"dataset_to_index":offset+length}]),
                        delta_indices=get_indices(timestamps,fps))
    source=np.arange(length*7).reshape(length,7)
    padding=0
    for t in range(length):
        indices,masks=query(obj,t+offset,0)
        actual,mask=sample_indices(length,t,50,offset)
        np.testing.assert_array_equal(actual,indices["u"])
        np.testing.assert_array_equal(mask,masks["u_is_pad"].numpy())
        np.testing.assert_array_equal(source[actual-offset],source[np.asarray(indices["u"])-offset])
        padding+=int(mask.sum())
    assert padding==chunk_statistics(length,50)["padded_action_steps"]


def test_official_feature_padding_images_masks_and_collation(tmp_path,upstream):
    import sys
    import torch
    info,robot,train=inputs(state_dim=3,action_dim=3,capacity=8)
    robot["actions"][0]["action.arm"]["subtract_state"]=True
    path=tmp_path/"robot.yaml"; path.write_text(yaml.safe_dump(robot),encoding="utf-8")
    c=LingBotTrainingContract.build(info,LingBotRobotConfigSchema.parse(robot,info["features"]),train)
    reference=upstream(path,SimpleNamespace(joints=["{'arm': 8}"],cameras=["top"]),None,None,do_nomalize=False,load_image=True)
    raw={"q":torch.tensor([1.,2.,3.]),"u":torch.ones(50,3)*4,"cam":torch.ones(3,4,5),"task":"pick",
         "u_is_pad":torch.tensor([False]*20+[True]*30)}
    converted=reference.apply(raw)
    batch=reference.pad_and_concat(converted)
    module=sys.modules[upstream.__module__]
    state=module.prepare_state(batch,32); action=module.prepare_action(batch,32); mask=module.prepare_joint_pad(batch,32)
    np.testing.assert_array_equal(mask.numpy(),c.to_dict()["joint_mask"]["values"])
    assert tuple(state.shape)==(32,) and tuple(action.shape)==(50,32)
    np.testing.assert_array_equal(action[0,:3].numpy(),[3,2,1])
    images,img_mask,_=module.prepare_images(None,batch,[8,8],image_keys=c.required_cameras)
    assert tuple(images.shape)==(1,3,8,8) and img_mask.tolist()==[True]
    root=Path(os.environ.get("LINGBOT_ROOT",PROJECT.parent.parent/"lingbot-vla"))
    collate=source_method(root/"lingbotvla/data/data_transform.py","VLADataCollatorWithPacking","__call__",
                         {"Sequence":list,"Dict":dict,"torch":torch,"default_collate":torch.utils.data.default_collate})
    item={"state":state,"actions":action,"images":images,"img_masks":img_mask,"joint_mask":mask,"action_is_pad":converted["action_is_pad"]}
    output=collate(SimpleNamespace(state_features=list(item)),[item,item])
    assert tuple(output["actions"].shape)==(2,50,32)
    assert tuple(output["action_is_pad"].shape)==(2,50)


def test_cli_report_is_separate_and_nonmutating(tmp_path,monkeypatch):
    from typer.testing import CliRunner
    from lerobot_cleaner.cli import app
    from lerobot_cleaner.training.compatibility import lingbot
    dataset=tmp_path/"dataset"; dataset.mkdir()
    robot=tmp_path/"robot.yaml"; robot.touch()
    train=tmp_path/"train.yaml"; train.touch()
    monkeypatch.setattr(lingbot,"check_training",lambda *a:{"compatible":False,"status":"incompatible","mutation_performed":False})
    result=CliRunner().invoke(app,["check-training",str(dataset),"--robot-config",str(robot),"--train-config",str(train)])
    assert result.exit_code==2
    assert "dataset_quality" in result.stdout and "training_readiness" in result.stdout
    assert list(dataset.iterdir())==[]


def test_clean_pipeline_training_failure_is_diagnostic(tmp_path,monkeypatch):
    from lerobot_cleaner.v30.pipeline import training_report
    from lerobot_cleaner.v30.v3 import V3Config
    from lerobot_cleaner.training.compatibility import lingbot
    monkeypatch.setattr(lingbot,"check_training",lambda *a:{"compatible":False,"status":"incompatible"})
    cfg=V3Config(training_check=options(tmp_path))
    assert training_report(tmp_path,cfg)["status"]=="incompatible"
    assert training_report(tmp_path,V3Config())["status"]=="not_requested"


def test_official_language_format_and_mask_helper(tmp_path, upstream):
    """Real prepare_language with a recording tokenizer double, NOT a runtime smoke."""
    import sys
    import torch
    calls=[]
    def tokenizer(prompt, **kwargs):
        calls.append((prompt,kwargs))
        return {"input_ids":torch.ones(1,8,dtype=torch.int64),"attention_mask":torch.tensor([[1,1,1,0,0,0,0,0]])}
    module=sys.modules[upstream.__module__]
    tokens,mask=module.prepare_language(tokenizer,{"state":torch.zeros(32),"prompt":["pick"]},8)
    assert calls[0][0]==["<bos>pick\n"]
    assert calls[0][1]["padding_side"]=="right" and calls[0][1]["truncation"]
    assert tuple(tokens.shape)==tuple(mask.shape)==(8,) and mask.dtype==torch.bool


def test_official_task_source_is_tasks_iloc_name():
    import torch
    root=Path(os.environ.get("LINGBOT_ROOT",PROJECT.parent.parent/"lingbot-vla"))
    getitem=source_method(root/"lingbotvla/data/vla_data/base_dataset.py","LeRobotDataset","__getitem__",{})
    rows=[{"episode_index":torch.tensor(0),"task_index":torch.tensor(0)}]
    tasks=pd.DataFrame({"task_index":[0],"task":["not used"]},index=["real instruction"])
    obj=SimpleNamespace(hf_dataset=rows,delta_indices=None,meta=SimpleNamespace(video_keys=[],tasks=tasks),image_transforms=None)
    assert getitem(obj,0)["task"]=="real instruction"


def test_video_interval_and_sampled_decode_failure(tmp_path):
    c=contract()
    c.raw_features["cam"]={"dtype":"video","shape":[4,5,3]}
    storage=Storage(tmp_path,{"fps":10,"features":c.raw_features})
    (tmp_path/"cam.mp4").write_bytes(b"synthetic existence only")
    storage.meta=SimpleNamespace(get_video_file_path=lambda *args:"cam.mp4")
    storage.episode_metadata=lambda *args:{"videos/cam/from_timestamp":5.,"videos/cam/to_timestamp":6.5}
    def decode(*args):
        raise ValueError("synthetic decoder failure")
    storage.video_frames=decode
    report=evaluate_storage(storage,c,options(tmp_path,decode_cameras=True,camera_sample_stride=5))
    result=report["episodes"][0]["cameras"]["observation.images.top"]
    assert not result["complete"] and result["checked_decode_frames"]==4
    assert result["available_frames"]<20 and not report["compatible"]


def test_training_config_paths_relative_to_cleaning_yaml(tmp_path):
    from lerobot_cleaner.v30.v3 import V3Config
    path=tmp_path/"clean.yaml"
    path.write_text("training_check:\n  robot_config: robot.yaml\n  train_config: train.yaml\n",encoding="utf-8")
    cfg=V3Config.from_yaml(path)
    assert cfg.training_check.robot_config==tmp_path/"robot.yaml"
    assert cfg.training_check.train_config==tmp_path/"train.yaml"


def test_training_checker_never_runs_smoke_implicitly(tmp_path, monkeypatch):
    from lerobot_cleaner.training.smoke import lingbot
    monkeypatch.setattr(lingbot,"smoke_preprocessing",lambda *args:pytest.fail("unexpected smoke"))
    c=contract(); report=evaluate_storage(Storage(tmp_path,{"fps":10,"features":c.raw_features}),c,options(tmp_path))
    assert not report["runtime_validated"] and report["training_ready"] is None
    assert report["smoke"]["status"]=="not_run"


def test_pinned_defaults_and_unforwarded_chunk_match_official_source():
    from lerobot_cleaner.training.contracts.lingbot import DEFAULTS
    root=Path(os.environ.get("LINGBOT_ROOT",PROJECT.parent.parent/"lingbot-vla"))
    path=root/"lingbotvla/utils/arguments.py"
    if not path.is_file(): pytest.skip("Official argument source unavailable")
    tree=ast.parse(path.read_text(encoding="utf-8"))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="TrainingArguments")
    for node in cls.body:
        if isinstance(node,ast.AnnAssign) and node.target.id in DEFAULTS:
            fields={kw.arg:kw.value for kw in node.value.keywords}
            value=ast.literal_eval(fields["default"] if "default" in fields else fields["default_factory"].body)
            assert DEFAULTS[node.target.id]==value
    entry=ast.parse((root/"tasks/vla/train_lingbotvla.py").read_text(encoding="utf-8"))
    calls=[n for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="VLADataset"]
    assert calls and all("chunk_size" not in [kw.arg for kw in call.keywords] for call in calls)


def test_actual_mapped_values_must_be_available(tmp_path):
    c=contract(); storage=Storage(tmp_path,{"fps":10,"features":c.raw_features})
    storage.frames[0].at[0,"u"]=np.array([float("nan")]*7)
    report=evaluate_storage(storage,c,options(tmp_path))
    assert not report["compatible"]
    assert report["episodes"][0]["representations"]["action.arm"]["nonfinite_samples"]==1


def test_profile_report_does_not_overwrite_training_contract(tmp_path,monkeypatch):
    import json
    from lerobot_cleaner.v30 import review_report
    monkeypatch.setattr(review_report,"provenance",lambda *a:{})
    readiness={"status":"incompatible","compatible":False,"findings":[]}
    report={"episodes":1,"frames":20,"tasks":1,"fps":10,"video_verification":"metadata_only","training_readiness":readiness}
    config=SimpleNamespace(model_dump=lambda **kwargs:{})
    review_report.write_review_bundle(tmp_path,report,config,config,project=tmp_path,training={"status":"blocked"})
    saved=json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))
    assert saved["training_readiness"]==readiness
    assert saved["training_runtime_preflight"]["status"]=="blocked"
