# 基于 canonical feature 的 quality groups

## 原有依赖与本轮边界

原先 QualityGroup 强制 source + columns；audit_trajectory 直接按列提取，calibration 报告也直接读取这两个配置字段。DROID、LIBERO 配置重复维护 arm/gripper 的固定数字范围。

现在 group 优先引用 CanonicalFeatureSchema 中的准确名称。各 quality rule 的数组接口不变，旧 source + columns、无 groups 的单源配置也继续有效。本轮只改 quality group 的选择机制；独立 gripper、static_trim、profile motion_action_dims 等配置仍维持原有契约。

## 三种配置

```yaml
quality:
  groups:
    - name: left_arm
      feature: observation.state.left_arm.position
      velocity: 1.0
      acceleration: 5.0
      jerk: 50.0
    - name: all_arms
      features:
        - observation.state.left_arm.position
        - observation.state.right_arm.position
      velocity: 1.0
    - name: legacy_arm
      source: state
      columns: [0, 1, 2, 3, 4, 5, 6]
      velocity: 1.0
```

这些名称是已有 canonical 名称，不是按字符串前缀猜测。LingBot YAML 可产生 observation.state.arm.position；通用 mapping 示例产生 state.arm。必须引用对应 schema 实际声明的名称。阈值仅为语法示例，需按数据单位选择。

也接受 `source: {feature: state.arm}` 或 `source: {features: [state.left, state.right]}`，解析后统一为顶层 feature/features。三种选择方式互斥，不允许混用 feature 与 columns 或 feature 与 features，也不允许静默覆盖 source。

## 解析职责与顺序

```text
CanonicalFeatureSchema
  → FeatureResolver.resolve_selection
  → QualityGroupResolver
  → resolved source / columns
  → 原 velocity / acceleration / jerk 等数组规则
```

FeatureResolver 唯一负责 canonical 名称查询、模态确认、列偏移和维度计算。QualityGroupResolver 只选择 semantic/legacy 路径，添加 group 错误上下文和报告元数据，不重写 slice grammar。

多 feature 组合采用 schema 顺序，无论请求列表是否反写；报告另外保留原请求顺序。每个 feature 内部的 source concat 顺序保持 mapping 定义。普通 schema 按 canonical feature 宽度累计偏移；raw_vectors（例如现有 GR00T）使用实际原始 slice 位置，保留未命名间隔，不误当成紧密排列。raw-vector 别名导致所选列重叠时拒绝组合。旧 columns 严格保留配置顺序，例如 [4, 3] 不会排序。

同组不能混合 state/action；image/video/language 不能交给数值运动规则。还会检查 feature 不存在、名称重复/歧义、空选择、无效维度、非法 slice、选择越界和 schema 与实际数组宽度不一致。报错包含 group 名、feature 名和原因。

不带 adapter/schema 的旧 frame-only API 仅声明 observation.state/action（或显式配置的完整列）为完整向量，不凭空推断 arm/gripper 子特征。可向 audit_trajectory 显式传入 schema，或者使用正式 adapter 路径。

## 报告和校准

每个 episode 的 groups 项同时包含兼容旧格式的 source/columns，以及：

```yaml
group: all_arms
requested_features:
  - observation.state.right_arm.position
  - observation.state.left_arm.position
selection_mode: semantic
resolved:
  source: state
  columns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
  dimension: 14
  feature_order:
    - observation.state.left_arm.position
    - observation.state.right_arm.position
  ordering: canonical_schema
  layout: packed_features
  feature_ranges: ...  # 每个 feature 的 columns、dimension、底层 source slices
```

quality 报告通过现有 V3 audit、dry-run、输入/输出复核路径输出。profile review 从 pipeline 接收同一 canonical schema。calibration 使用已记录的解析结果，并拒绝混合不一致的布局；生成的 thresholds.yaml 保留 feature/features，避免退回手工维护数字列号。

## 配置迁移与兼容性

- droid_v3.yaml 和 droid_v3_referenced.yaml 使用现有 LingBot canonical feature 名称替换 group 的数字列号，不改变其阈值、group 名称、数值来源或列顺序。
- 原 libero_v3.yaml 和 profile 保留原有数字配置作为兼容入口。
- 新 libero_v3_semantic.yaml 配套 mappings/libero.example.yaml，将布局声明集中在 mapping 中，quality 只引用 state.arm、state.gripper、action.arm、action.gripper。与旧 LIBERO group 的 source/columns 和检查结果有回归对照。
- 本轮不改 policy 的 group 路径：例如 groups/state_arm/jerk 继续有效。

## 文件清单

新增：lerobot_cleaner/v30/quality_groups.py、tests/test_semantic_quality_groups.py、configs/mappings/libero.example.yaml、configs/cleaning/libero_v3_semantic.yaml、本说明。

修改：lerobot_cleaner/adapters/resolver.py、lerobot_cleaner/v30/quality.py、calibration.py、episode_review.py、pipeline.py；configs/cleaning/droid_v3.yaml、droid_v3_referenced.yaml；tests/v30/test_quality_groups.py、tests/test_generic_mapping.py；README.md、configs/README.md。

## 统一测试结果

本轮统一运行 222 项：220 通过，2 项因环境缺少官方 LeRobot 0.4.2 而失败。新增 31 项语义 group 回归全部通过，包含 canonical 顺序、布局偏移变化、legacy 等价、raw-vector 间隔、跨模态错误、校准结果追踪、DROID 配置迁移以及 LIBERO 新旧配置对照。通用 mapping 的 V3 audit 接线和 observer schema 传递用例也通过。

受阻的测试是 test_group_audit_matches_across_engines_and_preserves_files、test_joint_check_is_readonly_and_matches_both_audit_engines；均在 import lerobot 阶段失败，未进入实际存储扫描。不能将上述单元/受控 storage 用例视为真实官方存储集成验证。本轮没有运行生产数据清洗。
