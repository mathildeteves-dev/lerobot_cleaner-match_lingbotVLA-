# LingBot Training Contract Check

## A. 官方训练 sample 与源码依据

本次审查固定 LingBot `4eb34b7693a0565c67433f8fac9c59a2e67eb60b` 和 LeRobot `v0.4.2`。
这不是对任意未来版本的保证。

| 官方位置 | 确认的行为 |
|---|---|
| `lingbotvla/data/vla_data/base_dataset.py:103`，`VLADataset` | 建立 FeatureTransform，官方 LeRobotDataset，源图像先 Resize 到 224×224 |
| 同文件 `:165`，`get_delta_timestamps` | 所有原始 action 来源均用 `[t/fps for t in range(chunk_size)]` |
| LeRobot `datasets/utils.py:879` | `round(delta_seconds * fps)` 转整数偏移 |
| LeRobot `datasets/lerobot_dataset.py:928` | `_get_query_indices` 钳到当前 episode 两端，同时生成逐来源布尔 padding mask |
| LingBot `base_dataset.py:96` | `task = meta.tasks.iloc[task_index].name`；不是任意 language/instruction 列 |
| LingBot `utils.py:356` | 从首个 action 来源的 `_is_pad` 取 action_is_pad；有效配置各来源 mask 相同 |
| LingBot `utils.py:308`，`pad_and_concat` | 按 training joints 顺序补齐各关节组容量；joint_mask 由真实 **action** 维度生成 |
| LingBot `transform.py:196/209/224` | state/action/joint_mask 再补到模型 max dimensions |
| LingBot `transform.py:133/237` | 图像处理和 mask；任务文本加 `<bos>`、换行，再 tokenizer 右侧 padding/truncation |
| LingBot `data/data_transform.py:102` | `VLADataCollatorWithPacking` 组合 batch |

参考：[LingBot base_dataset](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/data/vla_data/base_dataset.py)、
[FeatureTransform](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/data/vla_data/utils.py)、
[LeRobot query](https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/datasets/lerobot_dataset.py)。

一个训练 sample 最终包含：
- state：`[max_state_dim]`。
- actions：`[H, max_action_dim]`，包括当前 t，不是从 t+1 开始。
- images / img_masks：按配置相机顺序；Qwen processor 可产生 patch layout。
- lang_tokens / lang_masks：`[tokenizer_max_length]`。
- joint_mask：`[max_action_dim]`，True 表示真实 action 维度。
- action_is_pad：`[H]`，True 表示该时间槽请求超出本 episode。

默认 H=50，state/action padded dim=32，tokenizer_max_length=48，图像模型 resize=224×224，
均集中记录在 contract DEFAULTS，并有官方参数源码对照测试。

### 两个不能忽略的官方实现细节

1. `tasks/vla/train_lingbotvla.py:134–137` 设置了 `data.chunk_size`，但调用 VLADataset
   时没有传 `chunk_size`。因此这个入口的 **dataset H 仍为 50**。
   `models/auto.py:96` 却将 `train.chunk_size` 作为模型 n_action_steps。
   配置非 50 时，checker 报 horizon_mismatch。`explicit_dataset` 模式表示调用者确实显式
   传 constructor 参数；现有 cleaner preprocessing smoke 正是此模式。
2. `models/vla/pi0/modeling_lingbot_vla.py:1433` 的 forward 接收 action_is_pad，
   但该 loss 路径没有使用它，仅用 joint_mask 筛选维度。因此“被标记 padding”不等于
   “被从 loss 删除”。报告不把高 padding 自动解释为坏数据或无效 loss。

## B. 原 cleaner 的缺口

原 `v30/training_readiness.py` 是运行依赖/语义预检，且明确标记尚未实际加载批次；
`v30/lingbot/config.py` 检查 robot mapping 与容量；`scripts/lingbot_training_batch.py`
提供真实 preprocessing/collator smoke。它们都不是逐 episode 的训练契约诊断。
缺失项包括 action-horizon 感知、边界 padding 分布、官方 task 文本来源覆盖、逐 episode
相机可用性、训练 joint layout 与 mask 的统一报告，以及 dataset/model H 不一致诊断。

## C. 架构与文件

```text
Dataset Quality / Integrity → Cleaning / Writer / Finalizers → Clean Dataset
                                                              ↓
robot.yaml → shared LingBotRobotConfigSchema → Training Contract + train config
                                                              ↓
                                              TrainingCompatibilityCheck
                                                              ↓
                                     explicit real preprocessing/collator smoke
```

本次新增/修改（不包含上一轮尚未提交的 adapter grammar 改动）：

| 文件 | 职责 |
|---|---|
| `training/config.py` | 独立 opt-in 参数、entrypoint、warning 与解码配置 |
| `training/contracts/lingbot.py` | 默认值及来源、schema 联动、维度/joint layout/mask/采样契约 |
| `training/compatibility/chunks.py` | 精确索引模拟、逐集和全数据 padding 统计 |
| `training/compatibility/lingbot.py` | 官方 storage 上的只读 sample、task、相机及实际 feature 可用性诊断 |
| `training/smoke/lingbot.py` | 已清洗数据上的真实 preprocessing+collator，固定 level 2，不加载模型 |
| `training/` 及三个子目录的 `__init__.py` | 包边界 |
| `adapters/lingbot_config.py` | 共享 parser 增加显式诊断模式，允许缺相机进入 compatibility 报告；默认 adapter 仍严格校验 |
| `v30/v3.py` | `training_check` 配置及相对路径解析 |
| `v30/pipeline.py` | 写出/复查后运行诊断；dataset_quality 与 training_readiness 分开 |
| `v30/review_report.py` | 保留训练契约结果，旧环境预检另存 training_runtime_preflight，避免 LIBERO 报告覆盖新结果 |
| `cli.py` | 新增 check-training 和可选无模型 smoke |
| `scripts/smoke_lingbot.py` | 旧 smoke 在清洗后、preprocessing 前执行契约检查 |
| `tests/test_training_contract.py` | A–K、官方行为对照、CLI 和报告集成测试 |
| `tests/v30/test_smoke_levels.py` | 编排 fixture 更新、新增契约错误阻止 smoke 的测试 |
| `configs/cleaning/training_contract.example.yaml` | 可合并到实际 cleaning YAML 的配置示例 |
| `README.md`、`configs/README.md`、本文 | 使用入口、范围、源码及验证记录 |

Contract 复用同一份 robot grammar，没有第二套 robot YAML 解释器。
训练 images 的切片维度按官方 Resize 后的 224 宽规范化；来源仍是原始 dataset 的真实相机。

## D. Action chunk 精确计算

令 episode 全局范围为 `[S,E)`、当前局部 timestep 为 t、L=E−S：

```text
requested[k] = S + t + k                  k = 0,...,H-1
query[k] = max(S, min(E-1, requested[k]))
action_is_pad[k] = requested[k] < S or requested[k] >= E
```

边界外的 action 不是填零，而是重复最后一个 action；零填充用于模型维度。
不会读取下一 episode。每个原始 action source 使用同一 H；整个 episode 每行是一个候选 sample。
统计的是正常顺序采样的候选集，不模拟官方 `__getitem__` 出错后的随机重试/替换分布。

逐集统计通过 `valid(t)=min(H,L−t)` 计算，与逐 timestep 官方 mask 求和完全一致，
避免为所有 L×H 槽存储大矩阵。单样本 `sample_indices` 保留实际索引与 mask，供对照/诊断使用。

| L | H | requested slots | padded slots | slot padding ratio |
|---:|---:|---:|---:|---:|
| 20 | 50 | 1000 | 790 | 0.79 |
| 60 | 50 | 3000 | 1225 | 0.408333… |
| 10000 | 50 | 500000 | 1225 | 0.00245 |

正常非空 episode 总包含当前帧，fully_padded_samples=0。L=20 时，每个 sample 都部分 padding，
因此 sample-level padding ratio=1，与 slot-level 0.79 不同。

## E. 指标与判断边界

### Action chunk

逐集：episode_length、num_training_samples、total_action_slots、usable_action_steps、
padded_action_steps、action_chunk_padding_ratio、boundary_padding_steps/ratio、
fully_unpadded_samples、partially_padded_samples、fully_padded_samples、sample_padding_ratio。
可选 padding_by_timestep，顺序就是 t=0...L−1。

`action_chunk_padding_ratio = padded temporal slots / requested temporal slots`。
这里的 slot 是 action 时间步，不是 padded action 的数值维度。当前前向采样全部 time padding
都来自边界；non_boundary_padding_steps=0，不混入 missing sensor 或 dimension padding。

全数据：total_samples、total_action_slots、total_padded_action_slots、usable_action_steps、
总槽加权比例；mean/median/p95 是等权 **episode ratio** 分布；gt_0_25/gt_0_5/gt_0_75
只是统计计数，不触发删除。可配置 warning，但没有 exclude/downweight/TransformPlan。

### Task / cameras / representation

- task_text_available、task_text_missing_ratio、unique_task_count、empty_task_count；
  按 sample 记录 missing index、missing mapping、映射排序错误、null/空白/非字符串计数。
  empty_task_count 是含空白指令的 sample 数，不是 unique task 数。
- 相机 required/available/missing/extra、extra_dataset_sources、camera_availability_ratio、
  complete_episodes，及每集每相机的帧覆盖、解码抽查数和失败原因。
- 默认检查 metadata、路径、video interval 对齐和内嵌 image 是否存在，**不代表视频可解码**。
  `--decode-cameras` 用官方 decoder 抽查；stride=1 检查每个 timestep。未抽到的损坏不能声称排除。
- state/action 分别记录 raw_dim（不同源列总维度）、mapped_dim（有序映射后）、
  joint_layout_dim（各组容量之和）、padded_dim、expected_dim、layout 及 dimension mask。
  同时读取实际 episode feature，报告缺列、切片无法提取或非有限值。
- joint_mask 依据真实 action 维度；state 缺失组可以零填充，不能拿 state mask 替代 action mask。
- 超过组容量/模型容量即 error：官方负 F.pad 实际会截断，checker 不把悄悄丢维度算兼容。
- subtract_state 使用对应 mapped state；convert_from_state 的官方 marker 不自动生成来源。
  若 action 和 state 共用原始列，官方 delta 查询会把该 state 也变成 H×D，报
  state_action_source_rank_collision，不能只按原始 shape 宣称兼容。

### 两类报告互不代替

`dataset_quality` 保留独立质量结果；`training_readiness` 可以是 static_compatible、
incompatible、not_requested 或 unavailable。兼容失败不会改写质量判断或删除 episode。
`static_compatible` 时 runtime_validated=false、training_ready=null；实际 tokenizer、
normalizer、视频解码和 collation 仍需 smoke。运行时依赖/控制语义预检保留在独立字段。

## CLI 与清洗集成

```bash
lerobot-cleaner check-training /data/clean_v3 --target lingbot \
  --robot-config configs/robot_configs/droid_franka.yaml \
  --train-config configs/vla/droid_franka.yaml \
  --output /reports/training.json --padding-by-timestep
```

这是独立只读训练诊断，dataset_quality 会标记 not_evaluated，不能伪称数据质量已通过。
exit code：0=静态兼容（仍可能 warning），2=契约不兼容或请求的 smoke 未通过，1=读取/运行错误。
报告必须是 dataset 外的新文件。v2.1 输入通过 `--config` 的 converted_root 使用既有官方转换副本。

在 cleaning YAML 加入 `training_check`，写出后自动生成独立报告，示例见
[training_contract.example.yaml](../configs/cleaning/training_contract.example.yaml)。内部路径相对该 YAML。
audit-v3 启用同一配置时检查当前输入；clean-v3 检查修改后的输出。

真实 preprocessing smoke（不加载大型 VLA 权重）：

```bash
lerobot-cleaner check-training /data/clean_v3 \
  --robot-config /configs/robot.yaml --train-config /configs/train.yaml \
  --entrypoint explicit_dataset --smoke \
  --lingbot-root /code/lingbot-vla --norm-stats /stats/norm.json
```

要求真实 LingBot/LeRobot/processor 依赖和本地 tokenizer 文件、匹配的 norm stats。
复用 training_batch(level=2)，实际走 FeatureTransform(normalize=True)、图像/语言准备、
padding、mask、DataLoader、官方 collator；local_files_only，不下载模型，也不开始训练。
旧 `smoke-lingbot --level 2` 同样保留；新增契约位于其清洗后、compute_norm/实际批次前。
`--smoke` 不会被静态检查隐式调用。

## F. 测试与参考源码

新增专门训练检查测试：A–K，另包含官方入口 H 差异、任务实际来源、warning 不改数据、
缺相机仍可形成报告、相机时序缺失/解码错误、源列 rank 冲突、实际映射非有限值、
相对路径、CLI 报告分离、pipeline 诊断隔离、profile 报告不覆盖契约。

一致性测试调用本机未修改的官方 FeatureTransform、padding/image helpers，以及从官方源文件
AST 提取、原样编译的方法体：delta 构造、LeRobot get_delta_indices/_get_query_indices、
任务来源、collator。这样避免为了一个纯采样方法导入大型训练依赖，但没有重写参考算法。
覆盖 fps=10/29.97、episode 起始 offset≠0、首尾 timestep、实际 chunk 值及 mask。
语言 helper 使用真实 prepare_language + recording tokenizer double，仅验证格式/shape 调用约定，
不冒充真实 tokenizer smoke。编排测试也明确使用 doubles，不是运行时通过证据。

LeRobot 参考源码本次从官方 v0.4.2 下载至 Git 忽略的 outputs/official-reference：
- lerobot_dataset.py SHA256 `bef8f912c8be5503051f8368608562456b700eacfda4d91911d8bdf629fac393`
- lerobot_utils.py SHA256 `b1d32074e843880ef81436d7357bd1ac7b8d26ad9d30a0594c38620d36fae692`

重跑时用 LINGBOT_ROOT、LEROBOT_DATASET_SOURCE、LEROBOT_UTILS_SOURCE 指向对应官方源码。
缺参考源码/torch/einops 会明确 skip，不能算官方一致性通过。

```powershell
$env:PYTHONPATH = "$PWD/.verification-deps"
python -m pytest tests/test_training_contract.py tests/test_lingbot_conformance.py tests/test_episode_builder.py tests/v30/test_smoke_levels.py tests/v30/test_lingbot_configs.py -q --tb=short -p no:cacheprovider --basetemp outputs/training-contract-new-run
```

最终结果：**142 passed，0 failed，0 skipped**，其中训练契约测试 42 项。
另有 2 条现有 Typer/Click 弃用 warning，不是功能失败。
源码初次下载被网络环境阻断，经允许后取到官方文件；没有遗留依赖 skip 或失败。
未运行全项目 pytest，也未对真实数据执行完整 preprocessing smoke/训练。
现有环境未安装完整 LeRobot/LingBot 运行依赖；不能将这些单元/官方方法一致性测试宣称为
真实 dataset + tokenizer + preprocessing 已跑通。CLI 会保留这种状态区分。

## 语言分词兼容性

新增 `training_check.tokenization.enabled`（默认 false）及 `check-training --tokenize`。使用训练配置的 `model.tokenizer_path` 或显式覆盖，按官方 `prepare_language` 参数对全部去重任务分词，报告 token 数、实际有效 token 数和截断比例；只读取本地资源/缓存。未启用时报告 `not_run`，不能视为分词验证通过。详见 [语言检查说明](LANGUAGE_TASK_INTEGRITY_ZH.md)。
