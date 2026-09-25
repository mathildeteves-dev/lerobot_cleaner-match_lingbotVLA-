"""Visual schema/backend equivalence and V3 policy integration."""
from dataclasses import asdict
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from lerobot_cleaner.adapters.lerobot_v3 import LeRobotAdapter
from lerobot_cleaner.adapters.resolver import FeatureResolver
from lerobot_cleaner.adapters.schema import FeatureSchema, FeatureSlice
from lerobot_cleaner.adapters.lingbot_config import LingBotRobotConfigSchema
from lerobot_cleaner.storage.visual import VisualReader
from lerobot_cleaner.v30.quality import TrajectoryQualityConfig
from lerobot_cleaner.v30.quality_options import VisualCheck
from lerobot_cleaner.v30.policy import QualityPolicy
from lerobot_cleaner.v30.visual import visual_findings
from lerobot_cleaner.v30.planning import build_plan, video_findings
from lerobot_cleaner.v30.v3 import V3Config


KEY = "observation.images.top"
PIXELS = np.arange(6*8*3, dtype=np.uint8).reshape(6,8,3)


def fixture(tmp_path, dtype="image", cell=None, shape=(6,8,3), names=None):
    spec = {"dtype": dtype, "shape": list(shape)}
    if names is not None:
        spec["names"] = names
    features = {KEY: spec, **{k: {"dtype":"float32", "shape":[2]} for k in ("observation.state", "action")}}
    info = {"fps":10, "features":features, "total_tasks":1, "total_episodes":1, "total_frames":2}
    video = tmp_path / "fake.mp4"
    video.write_bytes(b"fake reference; decoder is stubbed")
    storage = SimpleNamespace(root=tmp_path, info=info,
        tasks=pd.DataFrame({"task_index":[0], "task":["pick cup"]}),
        video_references=lambda eid: [{"key":KEY,"path":video,"start":0.,"end":.2}],
        video_frames=lambda eid,key,times: np.stack([np.moveaxis(PIXELS, -1, 0) for _ in times]))
    config = V3Config(quality={"visual":{"decode":True,"blur_variance":1,"black_level":3,"min_brightness":1}})
    adapter = LeRobotAdapter(tmp_path, config, storage=storage)
    frame = pd.DataFrame({"episode_index":[0,0],"frame_index":[0,1],"index":[0,1],"timestamp":[0.,.1],"task_index":[0,0],
        "observation.state":[[0.,0.],[1.,1.]],"action":[[0.,0.],[1.,1.]]})
    if dtype == "image":
        frame[KEY] = [PIXELS.copy() if cell is None else cell for _ in range(2)]
    return adapter, adapter.from_frame(frame)


@pytest.mark.parametrize("dtype", ["image", "video"])
def test_semantics_resolver_and_legacy_camera_api(tmp_path, dtype):
    adapter, episode = fixture(tmp_path, dtype)
    visual = adapter.get_visual_features()[0]
    assert visual == adapter.get_camera_features()[0] == episode.visual_features[0]
    data = asdict(visual)
    assert data["semantic_modality"] == visual.modality == "visual"
    assert data["storage_dtype"] == dtype
    assert data["canonical_path"] == "visual.top" and data["source_key"] == KEY
    assert data["shape"] == (6,8,3)
    assert set(data) == {"feature_name","canonical_path","semantic_modality","storage_dtype","source_key","shape","sources"}
    # Resolving an already resolved feature is stable.
    assert FeatureResolver().resolve_visual(visual, adapter.info["features"]) == visual


def test_image_video_use_identical_pixel_checks_and_policy(tmp_path):
    findings = []
    for dtype in ("image", "video"):
        adapter, episode = fixture(tmp_path, dtype)
        record, report, plan = build_plan(episode, adapter, adapter.config)
        checks = record["checks"]
        assert checks["visual_integrity"]["passed"]
        assert checks["video_integrity"]["alias_of"] == "visual_integrity"
        assert checks["visual_integrity"]["metrics"]["cameras"][0]["sampled_frames"] == 1
        findings.append([checks[key] for key in ("blur","black_frame","brightness")])
        assert len(episode.df) == 2 and plan.source_frames == 2
    assert findings[0] == findings[1]


@pytest.mark.parametrize("kind", ["bytes", "path", "pil", "array"])
def test_image_storage_cells_without_video_metadata_or_decoder(tmp_path, kind):
    image = Image.fromarray(PIXELS)
    buf = BytesIO(); image.save(buf, format="PNG")
    path = tmp_path / "frame.png"; path.write_bytes(buf.getvalue())
    cell = {"bytes":buf.getvalue()} if kind == "bytes" else {"path":"frame.png"} if kind == "path" else image if kind == "pil" else PIXELS
    adapter, episode = fixture(tmp_path, cell=cell)
    def forbidden(*args):
        raise AssertionError("Image backend must not query video references or decoder")
    adapter.storage.video_references = adapter.storage.video_frames = forbidden
    found = video_findings(adapter.storage, episode, adapter.config.quality.visual)
    assert found["visual_integrity"]["passed"]
    np.testing.assert_array_equal(VisualReader(adapter.storage,episode).frames(episode.visual_features[0],[0])[0],PIXELS)


@pytest.mark.parametrize("cell,reason", [({"bytes":b"broken"},"identify image"),({"path":"missing.png"},"Invalid image reference"),
    (np.zeros((2,2,3),dtype=np.uint8),"resolution"),(np.full((6,8,3),np.nan),"Nonfinite"),
    ({"bytes":b""},"Empty image bytes"),({"path":"../outside.png"},"Invalid image reference")])
def test_bad_images_return_findings_without_mutation(tmp_path, cell, reason):
    adapter, episode = fixture(tmp_path, cell=cell)
    found = visual_findings(adapter.storage,episode,adapter.config.quality.visual,episode.visual_features)
    assert not found["visual_integrity"]["passed"]
    assert reason in found["visual_integrity"]["metrics"]["cameras"][0]["error"]
    assert not found["blur"]["metrics"]["evaluated"]
    assert len(episode.df) == 2


def test_chw_metadata_and_encoded_image_shape(tmp_path):
    adapter, episode = fixture(tmp_path, cell=np.moveaxis(PIXELS,-1,0),shape=(3,6,8),names=["channels","height","width"])
    feature = episode.visual_features[0]
    assert feature.shape == (6,8,3)
    np.testing.assert_array_equal(VisualReader(adapter.storage,episode).frames(feature,[0])[0],PIXELS)
    buf = BytesIO(); Image.fromarray(PIXELS).save(buf,format="PNG")
    episode.df[KEY] = [{"bytes":buf.getvalue()}]*2
    np.testing.assert_array_equal(VisualReader(adapter.storage,episode).frames(feature,[0])[0],PIXELS)


def test_derived_mixed_visual_concat_and_source_order(tmp_path):
    adapter, episode = fixture(tmp_path)
    other = "observation.images.video"
    adapter.info["features"][other] = {"dtype":"video","shape":[6,8,3]}
    feature = FeatureResolver().resolve_visual(FeatureSchema("joined", "visual", (
        FeatureSlice(other,5,8),FeatureSlice(KEY,0,2))),adapter.info["features"])
    assert feature.storage_dtype == "mixed" and feature.source_key is None and feature.shape == (6,5,3)
    np.testing.assert_array_equal(VisualReader(adapter.storage,episode).frames(feature,[0])[0],
        np.concatenate([PIXELS[:,5:8],PIXELS[:,:2]],axis=1))


@pytest.mark.parametrize("spec", [{"dtype":"float32","shape":[6,8,3]}, {"dtype":"image","shape":[0,8,3]},
    {"dtype":"image","shape":[8,3]}, {"dtype":"image","shape":[6,8,3],"names":["a","b","c"]}])
def test_visual_metadata_validation_has_target_source_context(spec):
    with pytest.raises(ValueError) as caught:
        FeatureResolver().resolve_visual(FeatureSchema("top","visual",camera_column=KEY),{KEY:spec})
    for text in ("top",KEY,"shape"):
        assert text in str(caught.value)


@pytest.mark.parametrize("start,end", [(0,9),(2,2),(-1,2)])
def test_visual_slice_validation(start,end):
    with pytest.raises(ValueError,match="slice.*out-of-bounds"):
        FeatureResolver().resolve_visual(FeatureSchema("top","visual",(FeatureSlice(KEY,start,end),)),{KEY:{"dtype":"image","shape":[6,8,3]}})


def test_config_alias_roundtrip_and_policy_alias():
    old = TrajectoryQualityConfig(video={"decode":True,"blur_variance":3})
    new = TrajectoryQualityConfig(visual={"decode":True,"blur_variance":3})
    assert old == new == TrajectoryQualityConfig.model_validate(new.model_dump())
    assert old.video is old.visual
    assert QualityPolicy(rules={"video_integrity":{"on_fail":"reject_episode"}}).rules["visual_integrity"].on_fail == "reject_episode"
    with pytest.raises(ValueError,match="not both"):
        TrajectoryQualityConfig(visual={},video={})
    with pytest.raises(ValueError,match="not both"):
        QualityPolicy(rules={"video_integrity":{},"visual_integrity":{}})
    with pytest.raises(ValueError,match="decode"):
        VisualCheck(black_level=3)


@pytest.mark.parametrize("policy_key", ["video_integrity","visual_integrity"])
def test_bad_image_rejection_uses_one_canonical_decision(tmp_path,policy_key):
    adapter, episode = fixture(tmp_path,cell={"bytes":b"corrupt"})
    adapter.config.policy = QualityPolicy(default={"on_fail":"report","on_unevaluated":"report"}, rules={policy_key:{"on_fail":"reject_episode"}})
    _, report, plan = build_plan(episode,adapter,adapter.config)
    assert plan.reject_episode
    assert [d["rule"] for d in report.decisions].count("visual_integrity") == 1
    assert not any(d["rule"] == "video_integrity" for d in report.decisions)


def test_lingbot_chw_width_slice_uses_shared_resolver():
    schema = LingBotRobotConfigSchema.parse({"images":[{"observation.images.joined":{"origin_keys":[{KEY:{"start":3,"end":8}}]}}]},
        {KEY:{"dtype":"image","shape":[3,6,8],"names":["channels","height","width"]}})
    assert schema.images[0].sources[0].end == 8
