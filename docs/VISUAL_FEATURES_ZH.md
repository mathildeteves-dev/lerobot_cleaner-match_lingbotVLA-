# Canonical visual feature

相机观测的语义统一为 `visual`。物理存储由 `storage_dtype=image/video` 表示，
不再由 `modality=video` 同时承担两个职责。普通 LeRobot、GenericMapping、LingBot、GR00T
均通过 `FeatureResolver.resolve_visual()` 构建相同的 `VisualFeatureSchema`。

## API 与解析结果

`adapter.get_visual_features()` 是语义接口；`get_camera_features()` 继续可用。
`CanonicalFeatureSchema.visual` 与兼容字段 `cameras` 指向相同声明；
`UnifiedEpisode.visual_features` 暴露该声明。

```yaml
feature_name: observation.images.top
canonical_path: visual.top
semantic_modality: visual
storage_dtype: image
source_key: observation.images.top
shape: [480, 640, 3]
sources:
  - source_key: observation.images.top
    storage_dtype: image
    shape: [480, 640, 3]
    layout: HWC
    start: null
    end: null
```

image/video 返回相同结构。旧 feature name 保持不变；canonical path 与存储格式无关。
尺寸统一为 HWC，metadata 的 `names` 可指定 HWC 或 CHW（兼容 channel/channels）；
没有 names 的旧数据沿用 HWC。image 不要求 video.info、MP4 路径或视频时间偏移。
无效 dtype、尺寸、轴名、宽度切片、拼接高度/通道及 canonical path 冲突会明确报错。

派生相机保留有序 `sources`，每个源携带独立存储类型、HWC 尺寸及宽度切片。
拼接目标没有单一物理键时 `source_key=null`；异构源使用 `storage_dtype=mixed`，
读取器按各源 dtype 分发后沿宽度拼接。该操作只生成分析数组，不修改原始帧。
类型允许未来扩展；当前不实现 depth 后端，也不会把 depth 猜测成 RGB。

## 质量配置

```yaml
quality:
  visual:
    enabled: true
    decode: true
    sample_stride: 30
    batch_frames: 32
    blur_variance: 4
    max_blur_ratio: 0.5
    black_level: 3
    max_black_ratio: 0.5
    min_brightness: 1
    max_brightness: 254
policy:
  rules:
    visual_integrity:
      on_fail: reject_episode
    black_frame:
      on_fail: warn
    brightness:
      on_fail: warn
```

像素阈值均为可选，不配置则不启用。默认只检查引用，不解码。
`decode=true` 时统一校验采样像素的有限性、实际尺寸与 metadata，并计算像素指标。
亮度为 RGB（或灰度）通道均值，黑帧为至少 98% 像素灰度不大于 black_level。
模糊度和黑帧比例按所有相机采样汇总，亮度范围要求每个采样均满足。
float 像素按官方 [0,1] 约定转换为 [0,255]；整型目前支持 uint8。
元数据检查并不证明图片可解码；采样检查也不等价于逐帧检查。

`VisualReader` 负责存储分发：视频使用现有官方 decoder；图片支持 HF bytes/path、
PIL 和数组。路径引用限制在数据集根目录内。图片读取不调用视频引用或 decoder。
core quality 只接收观察结果/像素，不操作文件。缺失、损坏或尺寸错误产生失败报告，
像素采样不完整时像素检查标为 unavailable，交给现有 policy 决策。

兼容旧 `quality.video`、`VideoCheck`、`video_findings()` 和 `video_integrity` policy。
序列化配置采用 `visual`；同时配置新旧名称会报冲突。
报告保留 `video_integrity` 别名（alias_of=visual_integrity），policy 只处理一次。
原完整视频解码、MP4 写入、ROI 与 finalizer 后端保留；完整视频校验失败归入
visual_integrity，同时更新旧报告别名。完整视频检查仍只针对物理视频，图片走新视觉入口。

## 本轮文件清单

- Schema / adapters：`adapters/schema.py`、`adapters/resolver.py`、`adapters/base.py`、
  `adapters/episode.py`、`adapters/__init__.py`、`adapters/lerobot_v3.py`、
  `adapters/groot.py`、`adapters/lingbot.py`、`adapters/lingbot_config.py`、`v21/legacy_reader.py`。
- 存储：新增 `storage/visual.py`、`storage/images.py`；`storage/writer.py` 复用图片解码函数，保留旧导入接口。
- 纯检查：新增 `core/quality/integrity/visual.py`、`core/quality/vision/pixels.py`；
  `core/quality/integrity/video.py` 保留兼容 wrapper，模糊度复用原 `vision/blur.py`。
- V3 接入：新增 `v30/visual.py`；修改 `v30/quality_options.py`、`v30/quality.py`、
  `v30/planning.py`、`v30/policy.py`、`v30/pipeline.py`、`v30/episode_review.py`。
- 测试与说明：新增 `tests/test_visual_features.py`、本文；更新 `README.md`。

统一回归共 248 项：246 通过（包含新增 26 项），2 项因环境缺少官方 `lerobot` 包失败。
两个失败均为已有跨引擎数据访问集成用例，在初始化官方存储时失败。
测试中的视频解码使用替身，因此不据此宣称已验证真实 MP4 解码或官方端到端写出。
