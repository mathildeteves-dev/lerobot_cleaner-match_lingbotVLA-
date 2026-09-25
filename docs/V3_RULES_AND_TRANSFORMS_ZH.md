# V3 质量检查、策略和数据修改

## 唯一执行链

```text
LeRobotDataset → Adapters / FeatureResolver → CanonicalFeatureSchema
    → EpisodeBuilder → UnifiedEpisode → read-only TrajectoryView
    → Quality / Integrity Checks
    → QualityReport + TransformPlan
    → Transforms（工作副本）→ Clean UnifiedEpisode
    → Dataset Writer → LeRobot v3 Dataset → Finalizers → 输出质量复查 → 发布
```

`run`、`audit-v3`、`clean-v3`、`calibrate-v3` 和 streaming 兼容入口使用同一个
`v30/pipeline.py`。memory/streaming 配置名称保留，但都逐 episode 评估，
没有绕过 plan 的旧数值清洗分支。官方 HF 初始化仍可能缓存完整数据集。

## 规则与职责

| 类别 | 检测 / 报告 | 实际修改 |
|---|---|---|
| R1 时间对齐 | timestamp / episode_structure | 显式 retime 或结构修改后的官方重建 |
| R2 静止 | static_ratio、joint_static_ratio、static_edges 的候选区间和 drop_frames | transforms/motion/static_trim + trajectory/frame_filter |
| R3 夹爪 | gripper 的范围/二值约定检查 | transforms/embodiment/gripper |
| R4 ROI | 原图尺寸与像素区域校验 | transforms/vision/roi_crop，由 writer 永久重写图像/视频 |
| R5 episode 长度 | episode_length | policy 生成 EpisodeDecision，EpisodeFilter 决定是否写入 |
| R6 数值 | finite、joint_limits、zscore、percentile_outlier、velocity/acceleration/jerk 及导数 Z-score | numeric/repair、numeric/clipping；frame_filter 或 EpisodeFilter |
| R7 视频 | 官方引用/时间区间、采样解码、可选全文件解码；blur 低细节启发式 | 视频重写只由 writer 执行，不由 checker 删除文件 |
| R8 收尾 | 输出质量与计数/索引/特征 schema 复查 | 官方 save_episode/finalize 重建 metadata、统计、分片和偏移 |

`core/quality` 全部只读。运动算法已有实现保留兼容导出，按 integrity/motion/
embodiment/vision 提供分组入口。`TrajectoryView` 数组维持不可写。
blur 是图像细节启发式，不等于确定的失焦；完整 PTS 检查与采样解码分别标明范围。
官方 loader 无法安全解析的数据布局错误仍属于加载失败，不能通过 policy 强行跳过。

## 配置

完整示例：`configs/cleaning/v3_quality_policy.example.yaml`。

- `quality`：只配置测量项目、维度和阈值。`groups` 中可配置 joint_limits、percentile_bounds、zscore。
  joint_limits 和 percentile_bounds 为标量或与选择维度一致的 low/high；不从机器人名称推测。
- `policy`：default 与 rules 分开。rule key 可写 `jerk` 或 `groups/arm/jerk`。
  允许 report、warn、reject_episode、abort；on_unevaluated 单独决定数据不足时如何处理。
  未启用的 check 不能绑定策略，未知规则名明确报错。
- `transforms`：只配置实际修改。静止裁剪、frame filtering、episode 拒绝、数值处理和 ROI 都在这里。
- 兼容的顶层 `nonfinite`、`bounds`、`aliases` 会生成明确的 NumericEdit，执行时不再直接改扫描中的 DataFrame。
  aliases 最后同步，避免修改 canonical 源后留下重复列不一致。

检测不通过不等于执行某个修复。若希望 clipping 修复越界，请把对应检查设为 report/warn，
并显式配置 clip；若设 reject_episode，则该 episode 不会进入 writer。
剩余非有限值没有修复/过滤计划时会阻止写出，报告说明具体列。
输出检查不会自动再次执行计划；output_on_fail=abort 会阻止发布。
输出阶段的 reject_episode 也阻止发布，避免在复查阶段偷偷第二次删集。

`percentile_clip` 可显式给出 low/high，也可省略两者并配置 quantiles。
后一种先对原始数据做一次独立只读参考统计，再生成固定边界的计划，避免边清洗边改变阈值。
使用固定种子的全源 reservoir；样本容量由 quantile_samples 指定，报告记录是否为近似估计。
写入配置保存解析后的固定 low/high，输出复查不会重新拟合阈值。

## 帧与视频对齐

TransformPlan 的 trim 和 drop_frames 都引用原始 episode-relative 行号。
它们合成一个 keep mask，一次性应用到 DataFrame 的所有列，包括 state/action、
images、depth 和其他 sensor。视频使用完全相同的 keep_indices 与原始 timestamp
向官方 decoder 查询，随后按保留顺序写入新视频。

删除内帧会改变采样时间语义，所以结构修改显式记录 reindex/retime。
官方 writer 按 fps 重建 frame_index、timestamp、全局 index、episode_index 和视频偏移。
原始 frame selection、来源 episode → 输出 episode 对照表保存在报告中。
不把“删除内帧后仍保留原时间间隔”冒充均匀采样。

## ROI 与 preprocessing

`purpose: dataset` 才裁剪输出像素，更新 height/width 并重新编码视频、生成新的元数据。
`purpose: preprocessing` 不改变数据集，只记录到 `model_preprocessing.json` 供训练流程使用。
这里不会自动修改 LingBot 训练配置或重复应用训练变换。
同一输出特征必须具有统一的图像尺寸。

## 写出与 Finalizers

有数据修改时，`storage/writer.py` 调用官方 LeRobotDataset.create/add_frame/save_episode。
所有保留 episode 重新写出；视频也会重新编码，不承诺字节级不变。
没有任何数据修改时复制原始标准组件，保留现有统计，避免无意义的重新编码。

Finalizers 调用官方 finalize，保留语义侧车文件，重新打开输出检查 episode 覆盖、
计数、重建后的索引/时间以及计划中的 feature shape/dtype。
LeRobot statistics 由官方写入流程重建；LingBot 训练归一化仍需实际训练 preprocessing 另行计算。

所有修改发生在新工作副本和临时目录。失败不发布半成品，也不删除源 parquet/video。
resume 检查源指纹和配置后从原始数据重新生成计划并重写，不复用半写入的官方 episode。
这取代旧的数值阶段/视频复制阶段续跑策略；旧 checkpoint 格式不兼容。

## Dry-run 和报告

`run --dry-run` / `audit-v3` 返回指标、QualityReport 和 TransformPlan，不调用 transform 或 writer。
v2.1 输入若尚未转换且配置了 converted_root，Storage 仍需先在副本完成官方转换；
需要完全不写文件的审计时，先提供已经转换好的 v3.0 数据。

cleaning_report 下包括 report.json、input_audit.json、transform_plans.json、
model_preprocessing.json 和使用的配置。清洗失败时，.partial 中保留输入报告与计划。
如果所有 episode 被拒绝，只保存报告，不发布空数据集。

本次按用户要求只编辑代码/文档，没有运行程序、测试、编译或 smoke；
因此这里描述的是已接入的代码路径，不代表已完成真实数据运行验证。
