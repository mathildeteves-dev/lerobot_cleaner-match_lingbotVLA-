# 项目状态文档 (PROJECT_STATUS.md)

## 未引用重复索引分片处理

新增显式 `metadata_referenced` 数据文件策略，默认仍为 strict。
只选择逐集元数据引用的文件，要求所选文件完整覆盖声明帧数，继续逐帧验证索引/演示/视频关系；
排除清单写入报告，不删除原文件。最终输出在 strict 策略下复核。
见 [重复分片处理步骤](docs/STALE_SHARDS_ZH.md)。最新测试：79 通过、6 因缺少 ffmpeg 跳过；Ruff 通过。

## 完整 DROID 分批支持

新增 PyArrow 分批读取、跨文件演示处理、增量全局统计、阶段级续跑、输出空间预检和进度显示。
DROID 默认配置启用 streaming，不再全量拼接数据；旧 memory 引擎保留用于小样例兼容。
本地 100 集真实数据已跑通分批清洗并核对内容；新增超过 100 万帧的合成读取测试。
全局分位数采用有界均匀抽样，逐集分位数精确计算；中断的数值阶段从头扫描，视频阶段可复用已验证副本。
详见 [服务器分批清洗步骤](docs/STREAMING_SERVER_ZH.md)。未直接运行服务器完整数据或 LingBot 训练。
最新回归：72 项通过、6 项因缺少 ffmpeg 跳过；Ruff 检查通过。

## 本分支补充（2026-09-04）

本次新增 DROID / Franka 的原生 v3 保守清洗和 LingBot 映射。真实样例：100 集、
32,212 帧、15 Hz、47 个任务、3 路共享 AV1 视频；19 集标记失败，默认仍保留。
全部数值字段无 NaN/Inf，本次输出未改变数值、未删帧；重算数值统计并校验视频复制哈希。
本次测试：48 通过、6 因缺少 ffmpeg 跳过；Ruff 检查通过。
详见 [DROID 使用说明](docs/DROID_GUIDE_ZH.md)。

当前验证边界：v3 数据结构/数值/映射静态检查及本地保守清洗已运行；
视频完整解码、官方 LeRobot 实际加载、LingBot 归一化/训练/真机控制尚未验证。
现有 R1Pro YAML 仅保留为未验证模板，不能用于当前 DROID 数据。
以下原有状态是上游 v2.1 项目的历史记录，不代表本分支已完成 LingBot 端到端训练。

## 当前代码结构（2026-09-18，按格式版本拆分）

```
lerobot_cleaner/                        # 项目根
├── configs/                            # 见 configs/README.md：cleaning / profiles / embodiments 手写源；
│   │                                   # robot_configs / vla 为 generate-lingbot-config 生成产物
├── scripts/                            # 各数据集的启动入口（按数据集命名，不进主包）
│   ├── run_droid_clean.py              # DROID v3 审计/清洗入口
│   ├── run_libero_clean.py             # LIBERO v3 审查清洗入口
│   ├── run_lingbot_norm.py             # 在 LingBot 源码目录启动 compute_norm
│   ├── run_lingbot_probe.py            # 用真实 LingBot VLADataset 抽样验证映射
│   ├── run_compute_norm_r1pro.sh       # R1Pro 归一化（未验证模板配套）
│   └── package_streaming_update.py     # 打服务器增量更新包
│
├── lerobot_cleaner/                    # 主包（按数据格式版本分组，cli.py 是唯一调度层）
│   ├── cli.py                          # typer CLI：v2.1（run/check/…）与 v3（audit-v3/clean-v3/
│   │                                   # export-lingbot/validate-lingbot/generate-lingbot-config）路由
│   ├── v21/                            # 旧 GR00T v2.1 清洗引擎（R1–R8），自包含，未被 v3 路径复用
│   │   ├── config.py / wizard.py / pipeline.py / parallel.py / types.py / report.py
│   │   ├── video_utils.py              # ffmpeg/ffprobe 封装（v2.1 视频规则用）
│   │   ├── reader.py / writer.py / stats.py / validate.py / inspector.py
│   │   └── rules/                      # checks / transforms / finalizers（R1–R8）
│   └── v30/                            # 新 LeRobot v3.0 + LingBot-VLA 路径
│       ├── v3.py                       # v3.0 元数据/数值/共享视频审计与保守清洗
│       ├── v3_streaming.py             # PyArrow 分批引擎（大数据集），可选 review 观察者
│       ├── v3_stream_stats.py          # 增量统计 + 固定容量分位数抽样
│       ├── episode_review.py           # 共享审查观察者 DatasetReview + task_texts（tasks.parquet 解析）
│       ├── video_review.py             # 共享视频抽样诊断（由 v3_streaming.verify_videos 调用）
│       ├── review_profile.py           # 数据集契约与质量阈值 profile 加载
│       ├── review_report.py            # 可移植自动审查报告束
│       ├── libero.py                   # LIBERO 专属入口 validate_libero（profile 驱动）
│       ├── training_readiness.py       # 训练就绪阻塞项检查（语义核实、LingBot 源码、依赖）
│       └── lingbot/                    # LingBot-VLA 集成子包
│           ├── vla.py                  # v2.1→v3.0 导出
│           ├── config.py               # robot/train 映射静态校验 validate_mapping
│           └── generate.py             # 从 embodiment spec 生成 robot+train 配置对
│
├── docs/                               # 按数据集/主题拆分的使用说明（DROID/LIBERO/分批/重复分片）
├── presets/                            # v2.1 时代 GR00T 清洗预设（R1Pro），与 v3 无关
└── tests/                              # 按版本拆分：v21/（test_config/rules/stats/validate/e2e）
                                        # 与 v30/（test_v3* / test_libero* / test_lingbot_* / test_review_*）
```

依赖方向约定：`v21/` 与 `v30/` 互不引用，仅 `cli.py` 同时路由两者。
`v30/` 内的 review 组件（`episode_review` / `video_review` /
`review_profile` / `review_report`）是 DROID 与 LIBERO 共用的 v3 审查层，
不依赖任何数据集专属模块；数据集专属逻辑只允许出现在 `libero.py`、
`scripts/run_*` 与 configs 中。`v30/lingbot/` 子包独立于清洗引擎，仅被 CLI 懒加载。

> lerobot_cleaner —— 面向 NVIDIA Isaac GR00T 后训练场景的通用 LeRobot 数据集清洗工具
>
> 本文档给"未来的自己 / 接手的协作者 / 想使用本工具的人"看。读完即可了解：项目做到什么程度、怎么用、每个文件的作用、后续可优化方向。
>
> 最后更新：2026-07-07

---

## 0. 一句话状态

**v1 已完成并在真实数据上验证通过**。工具可对任意维度的 GR00T-format LeRobot v2.1 数据集做模块化清洗，输出同时兼容 GR00T 与 pi05/openpi 两套训练框架（双格式 stats），且带完整的输入预检与清洗报告。20 个单元/端到端测试全过，ruff 干净。

- **项目位置**：`/mlp_vepfs/share/cyd/codebases/lerobot_cleaner/`
- **专用 conda 环境**：`lerobot_cleaner`（Python 3.10，已 `pip install -e ".[all]"`）
- **命令**：`lerobot_cleaner`（也接受 `lerobot-cleaner` 别名）

---

## 1. 项目能做到什么程度（能力清单）

| 能力 | 状态 | 说明 |
|------|------|------|
| 输入 | ✅ | 任意维度的 GR00T-format LeRobot v2.1（维数/臂数/夹爪数/视角数都不限，全从 modality.json 读，无硬编码） |
| 启动预检 `check` | ✅ | 一次性检查输入契约，把所有问题汇总成一条可操作报错；`run` 时自动预检 |
| 数据集摘要 `inspect` | ✅ | 维度/key/fps/长度分布一目了然 |
| 8 条清洗规则 R1–R8 | ✅ | 见第 3 节 |
| 双格式输出 | ✅ | 同时产出 `stats.json`(GR00T) 与 `episodes_stats.jsonl`(pi05/openpi)，已用 upstream lerobot 逐字逻辑验证 |
| 对齐不变量 | ✅ | 写完重开视频核验"视频帧数 == parquet 行数 == episodes.jsonl length" |
| 交互式向导 | ✅ | 无配置时逐条提问，保存成 yaml 供复用 |
| dry-run | ✅ | 秒级空跑，只出报告不写数据 |
| 断点续跑 `--resume` | ✅ | 跳过已写出的 episode |
| 清洗报告 | ✅ | md / json / 图 + 实际使用的配置副本 |
| Python API | ✅ | `from lerobot_cleaner import clean` |
| 预设 | ✅ | Galaxea R1Pro 双臂 / 单臂 |
| 并行 | ✅ | 进程池按 episode 并行 |

**已验证的真实数据**：
- R1Pro Tea（141 集, 16 维, fps15, AV1）—— dry-run 发现夹爪已二值化、每集有边缘静止帧
- R1Pro Cocktail 20260529（90 集, 16 维, 3 视角, fps15）—— 完整清洗跑通，双格式 stats 与 GR00T stats.json 数学一致（mean 最大差 0.00e+00），输出在 `/mlp_vepfs/share/cyd/datasets/test`

---

## 2. 怎么用（使用方式）

### 2.0 环境准备（首次）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate lerobot_cleaner
# 若是全新克隆：
cd /mlp_vepfs/share/cyd/codebases/lerobot_cleaner
pip install -e ".[all]"        # 需要 ffmpeg/ffprobe 在 PATH 里（视频规则用）
```

### 2.1 四个子命令

```bash
lerobot_cleaner check   <数据集>                     # 预检输入契约（推荐第一步）
lerobot_cleaner inspect <数据集>                     # 打印数据集摘要
lerobot_cleaner validate <配置.yaml>                 # 校验配置文件
lerobot_cleaner run     <数据集> [选项]              # 执行清洗
```

### 2.2 两种使用模式

**模式 A — 配置驱动**（熟练用户 / 批量复现）
```bash
# 自动发现数据集目录下的 cleaning_config.yaml
lerobot_cleaner run ./my_dataset
# 或显式指定
lerobot_cleaner run ./my_dataset --config ./my_config.yaml --output ./my_dataset_clean
```

**模式 B — 交互式向导**（首次使用 / 无配置）
```bash
lerobot_cleaner run ./my_dataset      # 无配置则进入向导，逐条提问后存成 yaml 并执行
```

### 2.3 推荐上手流程（三步）

```bash
# 1) 预检 + 看摘要
lerobot_cleaner check   /path/to/dataset
lerobot_cleaner inspect /path/to/dataset

# 2) dry-run 空跑（秒级，不写数据），看报告里每条规则触发了多少
lerobot_cleaner run /path/to/dataset --output /tmp/exp --dry-run --yes
cat /tmp/exp/cleaning_report/report.md

# 3) 拷贝示例配置改参数，正式跑
cp examples/cleaning_config.example.yaml my_config.yaml
#   编辑 my_config.yaml 打开/关闭规则、调参数
lerobot_cleaner run /path/to/dataset --config my_config.yaml
```

### 2.4 常用选项

| 选项 | 作用 |
|------|------|
| `--config, -c` | 指定配置文件 |
| `--output, -o` | 输出路径（默认 `<输入>_clean`） |
| `--preset, -p` | 套用预设参数（如 `galaxea_r1pro_dual_arm`） |
| `--dry-run` | 只出报告不写数据 |
| `--num-workers, -j` | 并行进程数（默认 8） |
| `--resume` | 跳过已写出的 episode |
| `--yes, -y` | 无配置时跳过向导用默认值 |

### 2.5 Python API

```python
from lerobot_cleaner import clean
report = clean("./my_dataset", "./my_dataset_clean", config="my_config.yaml")
report.print_summary()
```

### 2.6 输入要求（重要）

工具只清洗**已经是 GR00T-format LeRobot v2.1** 的数据集，不做原始数据转换，也不是通用 lerobot 清洗器（依赖 GR00T 专属的 `meta/modality.json`）。输入目录须形如：

```
my_dataset/
├── meta/
│   ├── info.json          # 需含 data_path, fps, chunks_size, features
│   ├── modality.json      # GR00T 专属：state/action key→{start,end}；video keys
│   ├── episodes.jsonl     # 每行 {episode_index, tasks, length}
│   ├── tasks.jsonl
│   └── stats.json         # 可选（输出会重算）
├── data/chunk-*/episode_*.parquet
└── videos/chunk-*/<video_key>/episode_*.mp4
```

parquet 必需列：`observation.state`、`action`、`timestamp`、`frame_index`、`episode_index`、`index`。`observation.state`/`action` 的向量宽度须与 modality.json 声明的最大 `end` 一致。用 `lerobot_cleaner check` 可提前发现所有不符项。

---

## 3. 清洗规则库（R1–R8）

每条规则在 yaml 里独立配 `enabled` 与参数。调用层显式分为 Checks（时间戳→输入长度→视频完整性→轨迹数值检测）、Transforms（数值修复→分位数裁剪→静止裁剪→夹爪二值化→ROI）、Finalizers（R8）。Checks 拒绝后不进入 Transforms；长度检查针对变换前的数据，不在裁剪后重复检查。Checks 仅记录问题或拒绝 episode，不修改数值或删帧。旧 numeric_sanity 配置展开为 trajectory 检测和 numeric 变换；NumericSanityRule 保留为兼容层。速度、加速度、jerk 默认关闭，启用须显式设置 limits，阈值单位为目标原始单位/秒的对应阶次。R8 在所有 episode 完成后统一运行，dry-run 跳过写出收尾。

| ID | 规则 (yaml key) | 作用 | 关键参数 |
|----|-----------------|------|---------|
| R1 | `timestamp_alignment` | 时间戳单调、dt≈1/fps、**均匀 dt 检查**、视频帧数 vs 行数 | `on_mismatch`(warn_only/strict_drop) |
| R2 | `static_frame_trim` | `trim_edges`(安全默认) 或 `drop_static_frames`(opt-in) | `mode`, `source`, `pos_threshold` |
| R3 | `gripper_binarize` | 夹爪维二值化；幂等（已二值化则跳过） | `targets`, `threshold`, `mode` |
| R4 | `video_roi_crop` | 按比例裁剪 + resize（ffmpeg 重编码，自动帧对齐） | `rois`, `resize_after_crop` |
| R5 | `episode_length_filter` | 过滤过短/过长 episode | `min_frames`, `max_frames` |
| R6 | `numeric_sanity` | NaN/Inf、关节限位、**百分位离群裁剪** | `on_nan`, `outlier_mode` |
| R7 | `video_integrity` | 视频可解码 + 各视角帧数一致 | `check_decodable` |
| R8 | `reindex_and_restats` | **永远最后且不可关闭**：重索引、重建 meta、重算双格式 stats、核验对齐 | `verify_alignment` |

**为何这些设计（贴 GR00T 真实加载器）**：
- 视频按整数帧号定位且硬断言"帧数==行数"→ R8 写完重开每个视频核验对齐
- GR00T 默认 min/max 归一化 → 单个离群帧会拉坏整体范围，故 R6 提供百分位裁剪
- action chunk 从连续行按 dt=1/fps 构建 → `drop_static_frames` 破坏均匀 dt，故降级 opt-in；writer 总是重建均匀时间戳
- stats 用 Welford + 蓄水池采样流式重算，内存有界

---

## 4. 文件位置及其作用（v1.0 历史记录，当前结构见上文）

```
lerobot_cleaner/                        # 项目根
├── pyproject.toml                      # 打包/依赖/命令入口/ruff/pytest 配置
├── README.md                           # 面向使用者的说明（含输入要求、规则表、双格式保证）
├── LICENSE                             # Apache-2.0
├── PROJECT_STATUS.md                   # ← 本文档
│
├── examples/
│   ├── cleaning_config.example.yaml    # 注释完整的配置样例（拷贝改用）
│   └── README.md
├── presets/
│   ├── galaxea_r1pro_dual_arm.yaml     # R1Pro 双臂推荐参数（16 维）
│   ├── galaxea_r1pro_single_arm.yaml   # R1Pro 单臂推荐参数
│   └── README.md
│
├── lerobot_cleaner/                    # 主包
│   ├── __init__.py                     # 暴露 Python API clean()
│   ├── cli.py                          # typer CLI 入口（inspect/check/run/validate）
│   ├── config.py                       # pydantic 配置 schema + 预设合并 + yaml 读写
│   ├── wizard.py                       # 交互式向导（questionary/rich）
│   ├── pipeline.py                     # 编排：预检→离群预扫→并行清洗→顺序落盘→核验→报告
│   ├── parallel.py                     # 单 episode 工作单元（跑规则+暂存产物，供进程池）
│   ├── types.py                        # EpisodeWork / VideoTransform 等共享数据结构
│   ├── report.py                       # 报告生成（md/json/matplotlib 图）
│   ├── video_utils.py                  # ffmpeg/ffprobe 封装（探测/解码/选帧重编码）
│   ├── py.typed                        # 类型标记
│   ├── dataset/
│   │   ├── reader.py                   # 只读加载 meta + ModalityResolver（dotted key→切片）
│   │   ├── inspector.py                # 扫描出 DatasetSummary
│   │   ├── validate.py                 # 输入契约预检（第 5 节新增能力）
│   │   ├── writer.py                   # 唯一写出组件：重索引/重建 meta/双格式 stats/核验对齐
│   │   └── stats.py                    # 流式全局 stats + 逐集 stats(compute_episode_stats)
│   └── rules/
│       ├── base.py                     # Rule 抽象基类
│       ├── __init__.py                 # 分阶段构建 + run_episode_stages()
│       ├── checks/integrity/           # timestamp.py (R1), episode_length.py (R5), numeric.py（R6 兼容层）
│       ├── checks/trajectory/          # finite / joint_limits / velocity / acceleration / jerk / outlier
│       ├── transforms/numeric/         # repair.py / percentile_clip.py
│       ├── checks/vision/              # video_integrity.py (R7)
│       ├── transforms/motion/          # static_trim.py (R2)
│       ├── transforms/embodiment/      # gripper.py (R3)
│       ├── transforms/vision/          # roi_crop.py (R4)
│       └── finalizers/                 # reindex_and_restats.py (R8)，调用 writer 完成写出
│
└── tests/                              # 20 个测试全过
    ├── conftest.py                     # 合成数据集 fixture（含 ffmpeg 生成的测试视频）
    ├── test_config.py                  # 配置 schema / 预设合并
    ├── test_rules.py                   # 各规则单元测试
    ├── test_stats.py                   # 流式 stats vs numpy / 逐集 stats
    ├── test_validate.py                # 输入预检
    └── test_e2e.py                     # 端到端：清洗合成数据集并核验所有不变量
```

---

## 5. 双格式输出（GR00T 与 pi05 双通）

清洗后的数据集**两套框架都能直接训练，无需再处理**（已核验双方加载器源码）：
- **GR00T** 读 `meta/stats.json`（mean/std/min/max/q01/q99）
- **pi05/openpi** 走上游 HF `lerobot` 加载器，v2.1 时读 `meta/episodes_stats.jsonl`（逐集 min/max/mean/std/**count**）

工具每次运行**同时输出两者**。逐集 stats 从已加载的 parquet 行算出，**不解码视频、零额外开销**；图像 key 故意省略（GR00T 自己也省、openpi 内部归一化图像、上游 `aggregate_stats` 取 key 并集）。已用 upstream lerobot 的 `_assert_type_and_shape` + `aggregate_stats` 逐字逻辑验证通过。

---

## 6. 接下来可优化的方向

按优先级/价值排列：

### 6.1 让其他人更好用（开源基本盘）
- [ ] **state/action 列名可配置**：目前硬编码 `observation.state`/`action`，对非 GR00T 但结构相似的数据集放开（低成本，扩大适用面）
- [ ] **发布到 PyPI**：`pip install lerobot-cleaner` 而非源码安装
- [ ] **README 增加 quickstart gif / 完整跑通示例数据**：新人零门槛体验
- [ ] **CLI 名字最终确定**：`lerobot-cleaner` 与官方 lerobot 略有混淆，可考虑更短的名字

### 6.2 规则与适配扩充
- [ ] **补更多本体预设**：LIBERO / DROID / Bridge / SO-100 等 GR00T 官方兼容本体
- [ ] **R4 视角裁剪的交互式拖框**：GUI 环境下用 `cv2.selectROI` 在首帧拖框选 ROI，自动转比例存 yaml（大纲已设计，未实现）
- [ ] **子任务标注 (sub_task) 支持**：GR00T loader 支持 sub_task 语言标注，清洗时可保留/重算
- [ ] **相机外参 / 多模态时间戳漂移检查**：社区后训练常见 case

### 6.3 性能与工程
- [ ] **AV1→H264 转码作为独立可选步骤**：顺带解决训练 step time 瓶颈（PROJECT_STATUS 记的优化项）
- [ ] **v2.1 → v3.0 自动升级**：GR00T 官方有脚本，可集成
- [ ] **预检时跳过 `count_frames` 探测**：跳过视频规则时不必探测帧数，省十几秒
- [ ] **更大规模数据的并行落盘优化**：当前落盘顺序执行以保证 index 精确，可评估分块并行

### 6.4 可观测性
- [ ] **报告增加更多可视化**：state/action 分布直方图、静止帧裁剪前后对比、ROI before/after（report.py 已留接口）
- [ ] **清洗前后 stats 对比表**：直观看清洗对分布的影响

---

## 7. 关键约定与陷阱

- **永不改动输入**：输出必须是新目录，`output == input` 会直接报错
- **R8 不可关闭**：它是唯一保证 meta 一致性的收尾步骤
- **视频规则对 AV1 很慢**：R7/R1/R4 会解码视频，AV1 编码下耗时显著；正式清洗挑无训练在跑时做（重 IO 会拖慢/阻塞训练）
- **跳过视频规则时视频直接复制**：不解码不重编码，保留原编码
- **对齐不变量是硬约束**：任何改变行数的规则都必须同步重切视频帧，R8 会核验
