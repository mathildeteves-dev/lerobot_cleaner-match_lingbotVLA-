# 三层数据流架构

## 1. Storage / Loader

```text
Raw Dataset
    │
    ├─ v2.1 → 官方转换器（副本）→ v3.0
    │
    ↓
LeRobotDataset（官方数据访问层）
    ↓
OfficialStorage
```

`lerobot_cleaner/storage/official.py` 是官方 abstraction 的边界。
所有统一入口默认且只接受 `reader_backend: lerobot`，依赖锁定的 LeRobot 0.4.2
（安装 `.[lingbot]`）；缺失依赖或加载失败直接报错，没有 native 回退。

- 数值数据：官方 `hf_dataset` 的 Arrow 视图，避免训练 `__getitem__` 附加变换。
- episode offset、info、tasks、stats：官方 `dataset.meta`。
- parquet/video 路径：官方 meta 路径方法；视频时间区间来自官方 episode metadata。
- 视频时间查询：官方视频解码器。完整 PTS 连续性检查通过 Storage 内的 PyAV 迭代器提供原始帧，
  质量层负责判断，不把视频采样后的时间误当作文件原始时间。
- chunk/shard 的物理布局和 schema：集中在 Storage。为无损写出读取文件头，
  不从文件头推断 episode，不在语义层直接读 parquet。

输出由官方 writer/finalizer 维护分片、字段、统计和视频；无变换时复制，变换时重写并原子发布。
这是写入职责；不把训练 `LeRobotDataset` 当作任意原地编辑器。
官方 HF 初始化可能加载/缓存整个数据集，`batch_rows` 仅限制 cleaner 写入缓冲，
不承诺限制官方 loader 的初始化内存。

## 2. Semantic Adapter

```text
                    LeRobotDataset / OfficialStorage
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
            robot.yaml        info / meta      modality.json
                │                 │                 │
         LingBotAdapter     LeRobotAdapter      GrootAdapter
                └─────────────────┼─────────────────┘
                                  ↓
                           FeatureResolver
                                  ↓
                       CanonicalFeatureSchema
                     (FeatureSchema + FeatureSlice)
```

适配器只解释语义，例如 `observation.state[0:7]` 对应哪些关节，
不负责找分片或计算 episode 文件偏移。robot.yaml 和 modality.json 是语义侧车文件，
不要求官方 LeRobot 理解它们；数据维度以官方 `meta.info` 为依据校验。

| semantic_adapter | 语义来源 |
|---|---|
| auto（默认） | robot_config 存在时选 LingBot；否则 modality_config 存在时选 GR00T；否则 LeRobot |
| lingbot | robot_config 的 states/actions/images、origin_keys、切片和拼接顺序 |
| lerobot | 官方 info.features 与 quality.state_column/action_column |
| groot | modality_config，或数据集 meta/modality.json，或转换归档的 source_modality.v21.json |

`auto` 不猜测机器人类型，也不会因为发现陌生 modality.json 就改变语义。
同时给出两份映射时必须显式选择适配器。
GR00T Schema 保留原始向量列号与未命名维度；LingBot 按 YAML 顺序拼接。
`subtract_state` 是训练元数据，清洗不会自动把绝对动作转成 delta。

`FeatureResolver` 校验特征名、模态、切片及需要减 state 的配对，生成统一 Schema。
`LeRobotV3Adapter` 是 `LeRobotAdapter` 的兼容别名，`Feature` 是 `FeatureSchema` 的兼容别名。

## 3. Episode Assembly / Quality

```text
CanonicalFeatureSchema
          ↓
EpisodeBuilder（EpisodeAdapter / EpisodeAssembler）
          ↓
UnifiedEpisode
          ↓
TrajectoryView
          ↓
Quality / Integrity → QualityReport + TransformPlan
          ↓
Transforms → Clean UnifiedEpisode → Dataset Writer → Finalizers
```

`adapter.read_episode()` / `from_frame()` 经 EpisodeBuilder 构建统一 episode。
纯数组质量规则处理 jerk、static、timestamp、joint limit、gripper 等；
视频质量规则从 Storage 提供的图像检查黑帧、低细节和静止画面，保持数值与视频各自的数据接口。
不把视频像素强塞进只承载 state/action/timestamps 的 TrajectoryView。

### 组装策略

`build(frame)` 接收官方 loader 提供的已经对齐的行，保留异常值供质量检查发现。
`build_streams(streams, timeline, episode_id=..., key=...)` 显式组装分开的数值流：

| 策略 | 行为 |
|---|---|
| alignment | exact / nearest / backward；非精确匹配必须配置 tolerance |
| key | timestamp（秒）或 frame_index；时间线必须有限、严格递增 |
| missing | preserve / error / zero / forward_fill；前向填充不回填起始缺失值 |
| pad_to | 尾部补至最小长度，不截断；仅用于 build_streams |
| padding | nan / zero / edge |

state/action 在参考时间线上配对；不跨 episode 匹配。
补值发生在数值视图中，保留 DataFrame 原始缺失证据。
`metadata["assembly"]` 包含 matched、source_rows 和 valid_mask。
默认 `to_trajectory()` 排除 padding；显式 `include_padding=True` 才包含填充行。
现有清洗入口不隐式重采样或 padding。直接 adapter.write_episode 仍保护身份行；
结构修改通过 TransformPlan、统一 frame filter 和官方 Dataset Writer 完成。
TrajectoryView 只读，检查不修改 episode。详见 [规则与变换](V3_RULES_AND_TRANSFORMS_ZH.md)。

## v2.1 与 v3.0

v3.0 直接打开。v2.1 在配置中指定一个源目录之外的新路径：

```yaml
reader_backend: lerobot
engine: streaming
converted_root: ../../converted/my_dataset_v3
semantic_adapter: groot
# 可选：不指定时使用随转换保留的 modality.json
# modality_config: ../robot_configs/my_modality.json
```

`prepare_dataset` 仅探测版本；随后调用现有的官方转换封装，对副本转换。
转换后的 modality.json 同时保留在 meta 和来源归档中。
转换标记记录源路径、元数据内容及文件大小/修改时间指纹；未变的源可复用副本。
源变化、转换目标被占用或缺少标记时明确拒绝覆盖，需更换 converted_root。
这不是全量数据内容哈希验证。

`converted_root`、`robot_config`、`modality_config` 的相对路径均以配置文件目录解析。
也可以先用 export-lingbot 显式转换，再把 v3.0 路径传入统一入口。

```bash
lerobot_cleaner run /data/source --config configs/cleaning/my_robot.yaml --dry-run
lerobot_cleaner run /data/source --config configs/cleaning/my_robot.yaml --output /data/cleaned
```

统一的 run、audit-v3、clean-v3、calibrate-v3 使用该版本策略；输出格式为 v3.0。
旧 v2.1 逐集 reader 和专用变换位于 `v21/legacy_reader.py`，旧 CLI 为 `run-v21-legacy`。
inspect/check 仍是明确面向 GR00T v2.1 的旧格式工具。legacy 路径不属于三层统一入口。
旧 `native` / `metadata_referenced` 配置不再受统一入口支持。
`droid_v3_referenced.yaml` 保留文件名但改成 strict 官方读取；应先单独修复残留分片。

## 本次修改的验证状态

按用户要求，只修改代码和文档，没有运行项目、转换器、测试、编译或 smoke。
旧测试中的 native reader、旧 GrootAdapter 和旧命令假设需要按新架构迁移；
历史测试通过记录不能代表本次重构已通过运行验证。
