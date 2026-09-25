"""Bundle orchestration tests: external runtime is replaced, never claimed validated."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from typer.testing import CliRunner
from lerobot_cleaner.cli import app
from lerobot_cleaner.adapters.schema import FeatureSchema, FeatureSlice, CanonicalFeatureSchema
from lerobot_cleaner.adapters.resolver import FeatureResolver
from lerobot_cleaner.adapters.lingbot_config import LingBotRobotConfigSchema
from lerobot_cleaner.v30.lingbot import bundle, norm
from lerobot_cleaner.v30.v3 import V3Config


@pytest.fixture
def environment(tmp_path,monkeypatch):
    source = tmp_path/"source"; (source/"meta").mkdir(parents=True)
    (source/"meta/info.json").write_text(json.dumps({"codebase_version":"v3.0","total_frames":2,"features":{
        "q":{"dtype":"float32","shape":[1]},"u":{"dtype":"float32","shape":[1]}}}))
    (source/"cleaning_report").mkdir(); (source/"cleaning_report/original.json").write_text('{"original":true}')
    robot = tmp_path/"robot.yaml"
    robot.write_text(yaml.safe_dump({"states":["q"],"actions":["u"],"images":[]}))
    train = tmp_path/"template.yaml"
    template = {"model":{"model_path":"organization/model","tokenizer_path":"organization/tokenizer"},
        "data":{"joints":[{"arm":1}],"cameras":["top"],"norm_type":"meanstd"},
        "train":{"chunk_size":50,"max_state_dim":2,"max_action_dim":2,"tokenizer_max_length":48,
                 "resize_imgs_with_padding":[224,224]}}
    train.write_text(yaml.safe_dump(template))
    root = tmp_path/"lingbot";root.mkdir()
    events = []
    monkeypatch.setattr(bundle,"lingbot_provenance",lambda p:{"commit":"test-commit","source_sha256":{"compute_norm.py":"fake"}})
    def clean(src,target,config):
        events.append("clean"); bundle.shutil.copytree(src,target)
    monkeypatch.setattr(bundle,"clean_v3",clean)
    def audit(target,config,**kwargs):
        assert kwargs["output_check"] is True
        events.append("validate")
        return {"quality_policy":{"abort_episodes":[],"rejected_episodes":[]},"warning":"ordinary quality warning"}
    monkeypatch.setattr(bundle,"audit_v3",audit)
    def check(target,options,config):
        events.append("compatibility")
        assert options.tokenization.enabled and options.decode_cameras
        assert options.entrypoint == "official_train"
        return {"compatible":True,"contract":{},"tokenization":{"evaluated":True},
                "findings":[{"severity":"WARNING","code":"high_boundary_padding"}]}
    monkeypatch.setattr(bundle,"check_training",check)
    def compute(root,target,robot,train,out,device):
        events.append("norm"); out.write_text('{"official_test_double":true}')
    monkeypatch.setattr(bundle,"run_normalization",compute)
    monkeypatch.setattr(bundle,"validate_norm_output",lambda *a:events.append("norm_validation"))
    def smoke(*args):
        events.append("smoke")
        return {"status":"passed","training_batch_validated":True,"processor_tokenizer_validated":True,
                "feature_transform_normalize":True,"model_loaded":False}
    monkeypatch.setattr(bundle,"smoke_preprocessing",smoke)
    return SimpleNamespace(source=source,root=root,train=train,robot=robot,out=tmp_path/"bundle",events=events,template=template)


def export(e,**kwargs):
    return bundle.LingBotBundleExporter().export(e.source,e.out,lingbot_root=e.root,train_config=e.train,robot_config=e.robot,**kwargs)


def test_complete_bundle_and_real_final_paths(environment):
    e=environment; result=export(e)
    assert result["training_ready"] and result["status"] == "ready"
    assert e.events == ["clean","validate","compatibility","norm","norm_validation","smoke"]
    assert {"dataset","robot_config.yaml","train_config.yaml","norm_stats.json","compatibility_report.json","cleaning_report","manifest.json"} <= {p.name for p in e.out.iterdir()}
    assert (e.out/"cleaning_report/original.json").read_bytes() == (e.source/"cleaning_report/original.json").read_bytes()
    train = yaml.safe_load((e.out/"train_config.yaml").read_text(encoding="utf-8"))
    assert train["data"]["data_name"] == "robot_config"
    for key,path in (("train_path",e.out/"dataset"),("norm_stats_file",e.out/"norm_stats.json"),("robot_config_root",e.out)):
        assert train["data"][key] == str(path)
    assert train["train"] == e.template["train"]
    assert result["artifacts_sha256"]["norm_stats.json"] == bundle.digest(e.out/"norm_stats.json")
    assert not (e.source/"norm_stats.json").exists()


@pytest.mark.parametrize("skip",["skip_clean","skip_smoke"])
def test_skip_controls_do_not_fabricate_readiness(environment,skip):
    result=export(environment,**{skip:True})
    if skip=="skip_clean":
        assert "clean" not in environment.events and "validate" in environment.events
        assert result["training_ready"]
    else:
        assert "smoke" not in environment.events and "norm" in environment.events
        assert result["status"]=="unverified" and not result["training_ready"]


@pytest.mark.parametrize("stage",["clean","quality","compatibility","norm","norm_validation","smoke"])
def test_failures_never_mark_ready(environment,monkeypatch,stage):
    e=environment
    def fail(*a,**kw): raise ValueError("injected failure")
    target={"clean":"clean_v3","quality":"audit_v3","compatibility":"check_training", "norm":"run_normalization",
            "norm_validation":"validate_norm_output","smoke":"smoke_preprocessing"}[stage]
    monkeypatch.setattr(bundle,target,fail)
    with pytest.raises(ValueError): export(e)
    manifest=json.loads((e.out/"manifest.json").read_text(encoding="utf-8"))
    assert not manifest["training_ready"] and manifest["status"]=="failed"
    assert not json.loads((e.out/"compatibility_report.json").read_text(encoding="utf-8"))["training_ready"]


@pytest.mark.parametrize("reason",["required_camera_missing","dimension_mismatch","horizon_mismatch","mapping_error"])
def test_contract_errors_gate_norm(environment,monkeypatch,reason):
    monkeypatch.setattr(bundle,"check_training",lambda *a:{"compatible":False,"findings":[{"code":reason,"severity":"ERROR"}]})
    with pytest.raises(ValueError,match="compatibility failed"): export(environment)
    assert "norm" not in environment.events


def test_smoke_status_alone_is_not_sufficient(environment,monkeypatch):
    monkeypatch.setattr(bundle,"smoke_preprocessing",lambda *a:{"status":"passed"})
    with pytest.raises(ValueError,match="smoke failed"): export(environment)
    assert not json.loads((environment.out/"manifest.json").read_text(encoding="utf-8"))["training_ready"]


def test_source_and_existing_output_protected(environment):
    e=environment
    e.out.mkdir();(e.out/"keep").write_text("safe")
    with pytest.raises(FileExistsError): export(e)
    assert (e.out/"keep").read_text(encoding="utf-8")=="safe"
    e.out=e.source/"inside"
    with pytest.raises(ValueError,match="disjoint"): export(e)


def test_template_requires_explicit_training_choices(environment):
    e=environment; del e.template["train"]["chunk_size"]
    e.train.write_text(yaml.safe_dump(e.template))
    with pytest.raises(ValueError,match="explicit train"): export(e)
    assert not e.out.exists()


def test_generated_robot_reuses_canonical_slices_and_parser(environment,monkeypatch):
    e=environment
    metadata={"q":{"dtype":"float32","shape":[2]},"u":{"dtype":"float32","shape":[1]},"cam":{"dtype":"image","shape":[4,6,3]}}
    visual=FeatureResolver().resolve_visual(FeatureSchema("top","visual",camera_column="cam"),metadata)
    schema=CanonicalFeatureSchema(states=(FeatureSchema("state.arm","state",(FeatureSlice("q",1,2),FeatureSlice("q",0,1))),),
        actions=(FeatureSchema("action.arm","action",(FeatureSlice("u",0,1),)),),cameras=(visual,))
    @contextmanager
    def adapter(*args): yield SimpleNamespace(info={"features":metadata},get_feature_schema=lambda:schema)
    monkeypatch.setattr(bundle,"v3_adapter",adapter)
    result=bundle.LingBotBundleExporter().export(e.source,e.out,lingbot_root=e.root,train_config=e.train)
    parsed=LingBotRobotConfigSchema.from_yaml(e.out/"robot_config.yaml",metadata)
    assert [s.start for s in parsed.states[0].sources]==[1,0]
    assert parsed.images[0].origin_key=="cam"
    assert (e.out/"canonical_schema.json").is_file()


def test_unnamed_vectors_are_not_guessed():
    schema=CanonicalFeatureSchema(states=(FeatureSchema("observation.state","state",(FeatureSlice("q",0,2),)),))
    with pytest.raises(ValueError,match="Cannot infer"): bundle.robot_document(schema)


def test_quality_policy_blocks_but_warning_does_not(environment,monkeypatch):
    monkeypatch.setattr(bundle,"audit_v3",lambda *a,**kw:{"quality_policy":{"abort_episodes":[0]}})
    with pytest.raises(ValueError,match="policy aborted"): export(environment)
    assert "norm" not in environment.events


def test_cli_registered_and_skip_smoke_exit_code(environment):
    e=environment
    result=CliRunner().invoke(app,["export-lingbot-bundle","--dataset",str(e.source),"--output",str(e.out),
        "--lingbot-root",str(e.root),"--train-config",str(e.train),"--robot-config",str(e.robot),"--skip-smoke"])
    assert result.exit_code==2, result.output
    assert not json.loads((e.out/"manifest.json").read_text(encoding="utf-8"))["training_ready"]


def test_norm_command_calls_official_script_and_single_process(tmp_path,monkeypatch):
    root=tmp_path/"lingbot";(root/"scripts").mkdir(parents=True)
    (root/"train.sh").touch();(root/"scripts/compute_norm.py").touch()
    monkeypatch.setattr(norm,"validate_mapping",lambda *a:{})
    monkeypatch.setattr(norm,"validate_norm_output",lambda *a:None)
    calls=[]
    def run(cmd,**kw):
        calls.append((cmd,kw));Path(cmd[-1]).write_text('{"count":2}')
    monkeypatch.setattr(norm.subprocess,"run",run)
    out=tmp_path/"norm.json"
    norm.run_normalization(root,tmp_path,tmp_path/"robot.yaml",tmp_path/"train.yaml",out,"2")
    command,kw=calls[0]
    assert command[:6]==["bash","-o","pipefail","train.sh","scripts/compute_norm.py",str(tmp_path/"train.yaml")]
    assert kw["cwd"]==root and kw["env"]["NPROC_PER_NODE"]=="1" and kw["env"]["CUDA_VISIBLE_DEVICES"]=="2"
    assert out.is_file()


def test_norm_validation_supports_optional_subtract_state(tmp_path,monkeypatch):
    (tmp_path/"meta").mkdir()
    (tmp_path/"meta/info.json").write_text(json.dumps({"total_frames":2,"features":{"action":{"dtype":"float32","shape":[1]}}}))
    robot=tmp_path/"robot.yaml";robot.write_text("actions: [{action: {origin_keys: action}}]\nstates: []\nimages: []\n")
    train=tmp_path/"train.yaml";train.write_text("train: {chunk_size: 50}\n")
    stats={"count":2,"norm_stats":{"action":{k:[0.] for k in ("mean","std","min","max","q01","q02","q98","q99")}}}
    out=tmp_path/"norm.json";out.write_text(json.dumps(stats))
    monkeypatch.setattr(norm,"validate_mapping",lambda *a:{"mapped_dimensions":{"action":1}})
    norm.validate_norm_output(out,tmp_path,robot,train)
