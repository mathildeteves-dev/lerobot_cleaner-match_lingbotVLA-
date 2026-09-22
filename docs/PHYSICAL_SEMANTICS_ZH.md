# Canonical physical semantics

FeatureSchema / normalized FeatureSpec 增加可选 unit、representation、control_type、periodic、coordinate_frame。
枚举定义在 core/physical.py；构造时验证枚举和严格布尔值。字段为 keyword-only，旧位置参数构造保持兼容。
不从 feature 名字猜语义，也不自动换算单位或坐标。

```yaml
canonical:
  state:
    arm.position:
      sources:
        - source: observation.joint_position
          start: 0
          end: 7
      unit: rad
      representation: joint_position
      periodic: true
    eef.position:
      source: observation.eef_position
      unit: m
      representation: cartesian_position
      coordinate_frame: robot_base
  action:
    eef:
      source: action
      representation: pose
      control_type: delta_pose
      coordinate_frame: eef
```

GenericMapping 接受目标级 metadata（source 或 sources 形式）。普通 LeRobot 默认整向量映射
保留 features 条目中显式给出的这五个字段；GR00T 可在映射 block 上声明。
LingBot 官方 YAML grammar 不扩展；没有声明的字段仍为 None。
混合单位/表示的向量应在 mapping 中拆成不同 canonical feature，不能给整个混合向量指定单一物理语义。

FeatureResolver 的 feature_ranges 记录 canonical_path、source_key、columns、dimension、sources
和五个物理字段。多源时 source_key 为空，sources 保留完整映射。
QualityGroupResolver 还输出与选中列一一对应的 physical_semantics，旧 source+columns 同样保留 metadata。
TrajectoryView 将 metadata 保存为不可变 tuple，跟随选择列重排；raw layout 的未命名列保持未知。
原始数值和时间戳不改变；重叠 raw alias 的物理声明冲突会报错。

PhysicalDeltaResolver 统一计算第一阶位移。periodic=true 的绝对角位置必须显式声明 rad 或 deg：
rad 使用 2π 周期，deg 使用 360 周期，输出区间 [-period/2, period/2)。
只能取得相邻采样间最短角位移；无法恢复多圈运动或超过半周期的真实运动。
速度以该位移除以实际时间间隔；加速度和 jerk 在后续阶次做普通差分，并沿用中点时间戳，绝不重复 wrap。
static_ratio、joint_static_ratio、static_edges 也复用该位移操作，避免边界跳变引发错误裁剪提案。

quaternion、rotation_matrix、pose/delta_pose、eef_pose 等尚无几何实现，相关差分检查
返回 evaluated=false 与原因，由 on_unevaluated policy 决策。periodic 缺单位或与控制类型不兼容也不猜测。
其余检查（finite、时间戳等）仍可正常运行。

旧 velocity/acceleration/jerk 规则名保持兼容，表示第 1/2/3 阶时间导数。
报告新增 derivative_order、interpretation、threshold_unit 与逐列物理声明：例如 joint_velocity
的一阶导数解释为 joint_acceleration，delta_joint_position 的导数解释为指令增量的变化率，
不声称它是实测速度。阈值仍按各列原生单位解释，不做归一化或自动选择机器人阈值。
没有 metadata 时保持原欧氏数值行为，并通过 physical_semantics_available=false 标识。

## 文件清单

- 新增 `core/physical.py`：枚举、PhysicalSemantics、统一 PhysicalDeltaResolver（兼容 Python 3.10）。
- `adapters/schema.py`、`mapping.py`、`resolver.py`、`lerobot_v3.py`、`groot.py`、`episode.py`、`__init__.py`：声明、配置解析、物理 metadata 传播和导出。
- `core/trajectory.py`：不可变逐列语义；`v30/quality_groups.py`、`v30/quality.py`：选择列时同步语义并写入报告。
- `core/quality/_derivative.py`：velocity/acceleration/jerk 及 derivative z-score 共用物理差分。
- `core/quality/static_ratio.py`、`joint_static_ratio.py`、`motion/static_edges.py`：静止判断复用同一操作。
- 新增 `tests/test_physical_semantics.py`（28 项）；本文与 `README.md`。

统一回归：274 项通过，2 项因缺少官方 lerobot 包失败；没有执行实际数据集清洗或写出。
