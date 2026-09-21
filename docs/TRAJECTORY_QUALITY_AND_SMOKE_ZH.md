# 共享轨迹检查与真实 LingBot smoke

## audit-v3（只读）

v2.1 EpisodeWork 和 v3 单 episode parquet 切片分别通过 V21Adapter、V30Adapter 转换为 core.TrajectoryView。
核心不依赖 LeRobot 版本、modality 或存储布局。内存和流式 audit-v3 均输出 trajectory_quality，每个 episode 含 checks。
不因质量告警删除行、修改值、裁剪视频或二值化夹爪。以下配置可合并进原有 v3 清洗配置：

```yaml
quality:
  enabled: true
  groups:
    - name: state_arm
      source: state
      columns: [0, 1, 2, 3, 4, 5, 6]
      velocity: null
      acceleration: null
      jerk: null
      velocity_zscore: 3.0
      acceleration_zscore: 3.0
      static_epsilon: 0.0001
    - name: action_arm
      source: action
      columns: [0, 1, 2, 3, 4, 5, 6]
      velocity: null
      acceleration: null
      jerk: null
      velocity_zscore: 3.0
      acceleration_zscore: 3.0
      static_epsilon: 0.0001
    - name: state_gripper
      source: state
      columns: [7]
      velocity: null
      acceleration: null
      jerk: null
      velocity_zscore: 3.0
      acceleration_zscore: 3.0
      static_epsilon: 0.0001
    - name: action_gripper
      source: action
      columns: [7]
      velocity: null
      acceleration: null
      jerk: null
      velocity_zscore: 3.0
      acceleration_zscore: 3.0
      static_epsilon: 0.0001
```

以上是 DROID 的 7+1 维布局，各组阈值互不继承，可独立调整；示例的 Z-score/epsilon 是待调参启发式，不是机械臂和夹爪共享的物理限值。
LIBERO 配置使用 state 0–5 / 6–7，action 0–5 / 6，不能照搬 DROID 列号。
分组报告位于 `trajectory_quality[].groups.<name>`，包含 source、columns、thresholds、checks；顶层 checks.finite 保留全轨迹检测，每组另有只针对选中分量的 finite。
review profile 同样支持 quality.groups，输出 trajectory_groups。机械臂检查不会因同帧夹爪异常而失败。
组名必须唯一、列号非负且不重复；越界明确报错。支持任意数量自定义组，不会自动猜测机器人维度或补入未配置的组。
旧 quality.source/columns 配置仍可使用，报告 mode=legacy_single_source；无 groups 时不会自动切换为两个全维度 source。
新分组配置不得同时设置非默认顶层 source/columns/阈值，以免误以为顶层阈值会被各组继承。
序列化时顶层默认值可保留用于旧配置往返；配置分组请始终把实际阈值写在组内。


absolute 阈值 null 表示只测量，threshold_applied=false；并非确认动作安全。
阈值必须按机器人单位/控制语义指定。static_ratio 是相邻 transition 的静止占比，epsilon 是原始单位每步位移容差。
导数按时间戳秒计算，Z-score 在每条轨迹的每个分量导数上计算 population mean/std（ddof=0），不是跨数据集统计。
常数导数与浮点微小残差计为零 Z-score；过短、非有限数据返回 evaluated=false，不能当作正常通过。
流式处理仍只缓存一个 episode 的数组，但逐 episode 指标汇总随 episode 数增长。
v2.1 numeric_sanity 也支持默认关闭的 velocity_zscore / acceleration_zscore（enabled / limits / on_violation 配置）。

```bash
lerobot-cleaner audit-v3 /data/source --config configs/cleaning/droid_v3.yaml
```

## NumericRepairRule 插值

默认 element-wise：只替换 NaN/Inf 所在分量；同帧健康分量完全保留。
NaN 与 Inf 各自遵循 on_nan / on_inf，不互相覆盖策略；支持点仅使用有限数值。
与此前行为一致，整个分量无有限支持时填 0，边界使用最近有效值；指标包含修复分量数和涉及帧数。
插值不更改行数或原始帧索引。

## smoke-lingbot：分级验证，无训练

在源码仓库根目录、LingBot 自己的 Linux/WSL 环境运行，并以 editable 方式安装本项目。
Python 3.12.3 为 preferred；其他 3.12.x 记录 warning 但允许继续，非 3.12 为 blocker。
仍要求 PyTorch 2.8.0、CUDA 12.8 且 CUDA 可用。报告在 environment.python_compatibility、warnings 和 blockers 中记录 Python 检查结果。
必须有真实源码、视频数据、机器人映射、训练配置和对应 profile，不使用 mock。

```bash
lerobot-cleaner smoke-lingbot /data/source \
  --output /data/smoke-new-run \
  --lingbot-root /path/to/LingBot-VLA \
  --profile /path/to/profile.yaml \
  --robot-config /path/to/robot.yaml \
  --train-config /path/to/train.yaml \
  --config /path/to/cleaning.yaml \
  --cuda-device 0 --level 2
```

输出目录须不存在且位于源数据集之外。默认 `--level 1` 保持旧行为；上例显式选择 Level 2。所有级别先执行 Level 1：
1. clean-v3（保留行与视频，启用视频完整验证）
2. validate-lingbot 映射验证
3. 官方 scripts/compute_norm.py（使用现有单 GPU 启动封装，验证产物）
4. 真实 VLADataset 构建与归一化
5. dataset[0]（跟踪索引请求，任何随机替代/重试均失败）
6. 真正 DataLoader 取 batch（num_workers=0；验证字段形状/有限值）

Level 2 继续执行真实的 processor/tokenizer 初始化，然后使用 `VLADataset(do_nomalize=True)`。
它内部的 `FeatureTransform.apply()` 完成 normalize → pad_and_concat → prepare_images → prepare_language，
使用官方 `VLADataCollatorWithPacking` 和 DataLoader（num_workers=0）生成训练形状的 batch。
读取官方 TrainingArguments 的字段默认值，但不实例化训练参数、不构造模型、不初始化训练器或分布式环境。
报告记录实际调用顺序，以及 state/actions/images、token 和各掩码的 shape/dtype；检查有限值、非空 token、
相机存在性和关节 padding 容量，避免容量不足时被负 padding 静默裁剪。
Qwen 图像允许官方 processor 的 patch 布局，不强制当成 BCHW。
processor/tokenizer 必须已在本地或缓存可用，离线加载，不自动下载。
当前 Level 2 支持 RGB VLA；深度对齐配置会明确失败，需要另外验证深度分支。

只有显式 `--level 3` 才进入可选的单次模型前向。它要求 train config 的 `model.model_path`
指向包含 safetensors 权重的本地完整 post-training checkpoint，不允许随机初始化或仅加载 VLM。
使用官方 build_foundation_model 加载，单 GPU、eval + no_grad，按配置使用 autocast；不执行 backward、
optimizer、FSDP 或训练循环。Level 2 的同一个 batch 送入一次 forward，并检查返回 tensor 的有限性。
模型及它依赖的 tokenizer/视觉资产都必须提前在本地准备好；缺失资产不会回退为随机模型或标记通过。

`smoke_report.json` 的 requested_level/completed_level 区分请求与真正完成的级别；
`loader_report.training_batch_validated` 和 `model_forward_validated` 分别表示 Level 2/3 的结果。
高级别失败时保留已经完成的低级别结果，总 status 仍为 failed，退出码非零。
passed 只覆盖请求级别，不代表已训练或控制语义已确认。
缺失依赖/版本/源码时返回 blocked 和非零退出码，且不开始复制数据。
真实集成测试 tests/v30/test_smoke_lingbot.py 通过环境变量 LINGBOT_SMOKE_DATASET、LINGBOT_SMOKE_OUTPUT、LINGBOT_ROOT、LINGBOT_SMOKE_PROFILE、LINGBOT_ROBOT_CONFIG、LINGBOT_TRAIN_CONFIG 启用；未配置则明确 skip，不视为通过。

本次本地环境是 Windows Python 3.13.9，缺少 datasets/lerobot，真实 LingBot smoke 尚未执行成功。


## 联合静止比例（joint_static_ratio）

`check_joint_static_ratio(trajectory, state_epsilon, action_epsilon, threshold=0.95)`
在每个原始相邻 transition 上分别计算全 state 与全 action 的最大绝对差，严格使用 `< epsilon`，
然后计算 `(state_static & action_static).mean()`。state/action 维数可以不同。
不是两个独立比例的比较、平均或最小值，也不根据 quality.groups 自动猜配对。

```yaml
quality:
  joint_static_ratio:
    state_epsilon: 0.0001
    action_epsilon: 0.0001
    threshold: 0.95
  groups: # 原有 groups 可继续配置，联合检测独立于这些分量诊断
    - name: state_arm
      source: state
      columns: [0, 1, 2, 3, 4, 5, 6]
```

结果在 episode 顶层 `checks.joint_static_ratio`（review 为 `trajectory_checks.joint_static_ratio`），
包括联合比例、联合静止 transition 数、两侧各自静止数、总 transition 数和阈值。
只有比例大于 threshold 才报 warning；等于 threshold 允许通过。
少于两帧、缺失任一侧分量、非有限值或差分溢出返回 evaluated=false，不伪装通过。
单组 static_ratio 保留为局部诊断，不能代替联合静止判定。
现有 long_static_state 仍是 state-only 的历史告警，不应解释为联合静止；联合结论请读取 joint_static_ratio。
所有检查仍只读，不删除帧，不触发变换。本次未执行真实 LingBot smoke。

## clean-v3 的输入与输出质量报告

memory 和 streaming 两个 engine 的 `cleaning_report/report.json` 统一使用：

- `trajectory_quality_input`：对原始 episode 检查，在插值、裁剪数值范围、同步 alias 前采集。
- `trajectory_quality_output`：重新读取写出的数据后检查，反映最终输出的实际数值。

两侧使用相同 quality 配置，保留 episode_index、groups、阈值、metrics 和联合静止检查。
clean 报告不再使用含义模糊的 `trajectory_quality`；独立 `audit-v3` 仍保留该字段。
设置 `quality.enabled: false` 时，两个字段均为 `[]`，表示没有执行检查。
无修改时两侧结果应一致；修复后应分别查看两侧结果，不以输出覆盖原始异常记录。
streaming 断点保存输入质量；恢复旧版缺少输入质量的断点时，会重新只读检查原始数据并补存，
不会将旧的空列表当作已完成的输入质量检查，也不会重复执行已完成的数值修复。


### 本地测试与真实 smoke 的区别

`tests/v30/test_smoke_levels.py` 使用测试替身验证参数传递、张量校验和分级边界，不算真实 LingBot 验证。
`tests/v30/test_smoke_lingbot.py` 是真实环境门禁；除了已有的路径环境变量，可用
`LINGBOT_SMOKE_LEVEL=2` 选择训练 batch，显式设为 3 才运行 forward（默认 1）。
本次实现仅读取本地 LingBot 源码并运行单元测试，未实际执行任何级别的真实 smoke。

## 两遍执行：Calibration → thresholds.yaml → Cleaning

先做只读校准（默认不会启动 cleaning）：

```bash
lerobot-cleaner calibrate-v3 /data/source \
  --config configs/cleaning/droid_v3.yaml \
  --output /data/droid-calibration
```

一次命令自动完成两遍：

```bash
lerobot-cleaner calibrate-v3 /data/source \
  --config configs/cleaning/droid_v3.yaml \
  --output /data/droid-calibration \
  --clean-output /data/droid-clean
```

也可先审查校准报告，再单独执行：

```bash
lerobot-cleaner clean-v3 /data/source \
  --config /data/droid-calibration/thresholds.yaml \
  --output /data/droid-clean
```

两个输出目录必须不存在、位于源数据集之外，且彼此不嵌套。首次 calibration 不修复数值、不删 episode、不改原文件；
即使原配置包含 interpolate 或 bounds，第一遍仍只走 audit。memory/streaming 使用同一个估计器。
必须显式配置并启用 quality.groups；不会猜测机器人维度，也不会把 state/action 或 arm/gripper 混成同一个总体。
DROID 的四组名称及列号沿用随附配置；LIBERO 用自己的布局。

### 统计口径与默认公式

对每个 group，分别估计 velocity、acceleration、jerk、velocity_zscore、acceleration_zscore。
每个 episode 提供一个该组的峰值样本：前三项为 max(abs(derivative))，后两项为已有检查中的 max_zscore。
导数沿用真实时间戳（秒）；Z-score 仍是在 episode 内、按分量计算。
对选中数据集全部 episode 的这些峰值精确计算，不做 reservoir 抽样，episode 等权：

```text
Tq   = quantile(episode_peaks, 0.995, method="linear")
m    = median(episode_peaks)
MAD  = median(abs(episode_peaks - m))
Tmad = m + 8 × 1.4826 × MAD
T    = max(Tq, Tmad)
```

因此 Q99.5% 指 **episode 峰值分布**，不是将所有 transition、所有关节混在一起后的 99.5% 分位数。
长 episode 不因帧数多而获得更高统计权重。当前阈值是数据驱动的质量告警阈值，不是验证过的机械臂物理安全限值。
`max` 选择更宽松的一侧，符合偏保守告警策略，但不会保证剔除所有异常或证明数据无异常。

可调参数：`--quantile 0.995`、`--mad-k 8`、`--min-samples 20`。
每组每项至少需要 min-samples 个有效 episode。低于一条期望上尾样本时（默认分位数约需 200 条）会提醒尾部分位数估计不稳。
MAD=0 时仍计算 quantile；若最终值为 0，则显式应用 1e-12 的数值下限以满足现有阈值必须大于 0 的接口，并在报告标记 floor_applied。
这一下限不是自动估计出的物理限值。Python API 的 CalibrationConfig 可调整 positive_floor 和 mad_scale。

短轨迹、非有限值、无效导数和无效 Z-score 不混入统计，每组每项记录排除数量和原因。
已有阈值判为失败但指标可计算的 episode **仍参与校准**，避免按旧阈值筛选造成偏差。
任一组/指标样本不足或统计溢出，保存报告、退出码非零，不生成 thresholds.yaml，不进入 cleaning。

### 产物与兼容性

- `calibration_report.json`：统计口径、参数、source/columns、有效与排除数量、median、MAD、Tq、Tmad、最终阈值、警告和后续 cleaning 状态。
- `audit.json`：第一遍只读检查的逐 episode/group 原始指标，可追溯阈值来源。
- `thresholds.yaml`：校准成功才生成；是完整可加载的 V3Config，保留原 aliases、bounds、engine 等设置，仅替换五类 quality 阈值。

YAML 继续使用现有 **groups 列表 + 标量阈值** 接口，例如 `groups[].velocity`，不是另起 `groups.state_arm.velocity.threshold` 字典接口。
static_epsilon、static_ratio、joint_static_ratio 及原始数值 bounds 都不会被此校准重新估计。
第一遍读取的源文件身份（info 内容以及文件路径/大小/mtime）会在扫描结束和自动 cleaning 开始前核对；这不是全文件内容哈希，期间仍须保持源数据不可变。
streaming 的帧数据读取有批量边界；精确统计保留逐 episode 指标，内存会随 episode 数量增长，不宣称常量内存。

v3 Cleaning Pass 继续保留全部行：生成阈值作用于 `trajectory_quality_input/output`，不自动删除 episode、不把速度/jerk 阈值当作原始动作值的 clipping bounds。
若以后需要按质量拒绝 episode，需要另外明确接受策略；当前不会隐式更改此前的 row-preserving 约定。
