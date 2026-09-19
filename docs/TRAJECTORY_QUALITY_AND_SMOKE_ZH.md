# 共享轨迹检查与真实 LingBot smoke

## audit-v3（只读）

v2.1 EpisodeWork 和 v3 单 episode parquet 切片分别通过 V21Adapter、V30Adapter 转换为 core.TrajectoryView。
核心不依赖 LeRobot 版本、modality 或存储布局。内存和流式 audit-v3 均输出 trajectory_quality，每个 episode 含 checks。
不因质量告警删除行、修改值、裁剪视频或二值化夹爪。以下配置可合并进原有 v3 清洗配置：

```yaml
quality:
  enabled: true
  state_column: observation.state
  action_column: action
  source: state
  columns: null
  velocity: null
  acceleration: null
  jerk: null
  velocity_zscore: 3.0
  acceleration_zscore: 3.0
  static_epsilon: 0.0001
  static_ratio: null
```

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

## smoke-lingbot：真实执行，无训练

在源码仓库根目录、LingBot 自己的 Linux/WSL 环境运行，并以 editable 方式安装本项目。
按本次约定预检 Python 3.12.3、PyTorch 2.8.0、CUDA 12.8 且 CUDA 可用。
必须有真实源码、视频数据、机器人映射、训练配置和对应 profile，不使用 mock。

```bash
lerobot-cleaner smoke-lingbot /data/source \
  --output /data/smoke-new-run \
  --lingbot-root /path/to/LingBot-VLA \
  --profile /path/to/profile.yaml \
  --robot-config /path/to/robot.yaml \
  --train-config /path/to/train.yaml \
  --config /path/to/cleaning.yaml \
  --cuda-device 0
```

输出目录须不存在且位于源数据集之外。依次执行：
1. clean-v3（保留行与视频，启用视频完整验证）
2. validate-lingbot 映射验证
3. 官方 scripts/compute_norm.py（使用现有单 GPU 启动封装，验证产物）
4. 真实 VLADataset 构建与归一化
5. dataset[0]（跟踪索引请求，任何随机替代/重试均失败）
6. 真正 DataLoader 取 batch（num_workers=0；验证字段形状/有限值）

smoke_report.json 保存阶段结果；passed 仅表示数据加载链路通过，不代表已训练、模型前向验证或控制语义已确认。
缺失依赖/版本/源码时返回 blocked 和非零退出码，且不开始复制数据。
真实集成测试 tests/v30/test_smoke_lingbot.py 通过环境变量 LINGBOT_SMOKE_DATASET、LINGBOT_SMOKE_OUTPUT、LINGBOT_ROOT、LINGBOT_SMOKE_PROFILE、LINGBOT_ROBOT_CONFIG、LINGBOT_TRAIN_CONFIG 启用；未配置则明确 skip，不视为通过。

本次本地环境是 Windows Python 3.13.9，缺少 datasets/lerobot，真实 LingBot smoke 尚未执行成功。
