# 通用 LeRobot feature mapping

默认 LeRobotAdapter 使用 quality.state_column / action_column 指定的单一完整向量，默认是 observation.state / action；不推测其他列的机器人语义。现有 LIBERO、DROID 的默认列名与拼接结果保持不变。

复杂布局在清洗配置中显式指定：

```yaml
semantic_adapter: generic
mapping_config: ../mappings/generic.example.yaml
```

只设置 mapping_config 时 auto 也会选择 GenericMappingAdapter。路径相对于清洗配置文件解析；与 robot_config / modality_config 同时存在时必须明确选择，禁止悄悄忽略通用 mapping。

mapping 文件：

```yaml
canonical:
  state:
    arm:
      - source: left_arm.position
        start: 0
        end: 7
      - source: right_arm.position
    gripper:
      - source: observation.gripper
  action:
    arm:
      - source: action.arm
    gripper:
      - source: action.gripper
```

解析为 state.arm、state.gripper、action.arm、action.gripper。目标可使用短名称或带对应 state./action. 前缀的全名；规范化后的重复名称报错。sources 严格按列表顺序拼接，目标按 YAML 顺序保留。直接映射也支持 `arm: observation.joint_position` 或 `arm: {source: observation.joint_position}`。省略 start/end 时使用完整向量，单独省略一个端点时从 0 开始或到实际维度结束。

需要断言拼接维度时使用：

```yaml
canonical:
  state:
    arm:
      dimension: 14
      sources:
        - source: left_arm.position
        - source: right_arm.position
```

普通配置使用严格非负边界；空切片、反向切片、越界、重复目标、缺失来源、非数值向量、无序/空 source 列表、未知字段均报错。错误包含 target、source、slice、actual dimension 和原因。运行时还检查源向量实际宽度与 metadata 是否一致，防止只截取前几维掩盖坏数据。

共享类型为 FeatureSpec / SourceSpec；FeatureResolver 负责源查找、数值源校验、切片、维度推断、rename、按序 concat 和运行时提取。LeRobot 默认映射与 GenericMappingAdapter 均使用此层。LingBot grammar 先解释 origin_keys、重复键别名和官方 Python 切片规则，再调用共享 resolver；subtract_state、convert_from_state 和图片 CHW 语义仍留在 LingBot integration。通用 resolver 不导入或解释模型语义。

CanonicalFeatureSchema 保持 states/actions 多特征声明；`episode.feature_arrays("state")` 提供保留语义名称的只读数组。只有现有 `to_trajectory()` 分析接口才按 schema 顺序拼成数组，兼容已有 quality groups 的位置索引。GenericMappingAdapter 默认保留官方 image/video 相机声明，语言关联和 storage/writer 路径沿用现有实现。此轮不新增通用图片裁剪 grammar。

依赖边界通过延迟导入保持：通用适配器和 FeatureResolver 不需要 LingBot config、LingBot 库、Qwen 或 transformers。官方存储访问仍使用 LeRobotDataset；mapping 只定义语义，不增加另一套 parquet/video loader。

新增回归覆盖严格切片、拼接顺序、维度声明、错误信息、命名数组、工厂选择、配置相对路径、V3 audit 接线，以及禁止导入模型 integration 的独立进程检查。统一测试结果见本轮交付说明。

## 本轮文件清单

新增：
- `lerobot_cleaner/adapters/mapping.py`：通用 spec 与 YAML grammar。
- `lerobot_cleaner/adapters/generic.py`：GenericMappingAdapter。
- `configs/mappings/generic.example.yaml`：完整配置示例。
- `tests/test_generic_mapping.py`：24 项通用映射与集成回归。
- `docs/GENERIC_MAPPING_ZH.md`：本说明。

修改：
- `adapters/resolver.py`、`adapters/schema.py`：共享解析、提取与源维度声明。
- `adapters/lerobot_v3.py`：默认 mapping 共用 resolver；保留 image/video 相机。
- `adapters/lingbot.py`、`adapters/lingbot_config.py`：归一化为通用 spec，保留官方专属语义。
- `adapters/episode.py`：命名 feature 数组接口。
- `adapters/factory.py`、`adapters/__init__.py`：通用入口与模型 integration 延迟导入。
- `v30/v3.py`：mapping_config 和 generic 选择、路径解析与冲突校验。
- `storage/official.py`、`pyproject.toml`：独立 `.[lerobot]` 安装入口；保留旧 `.[lingbot]` 别名。
- `README.md`、`configs/README.md`：使用入口。

上述 adapters/、v30/、storage/ 均位于 lerobot_cleaner/ 下；其他此前未提交修改保持不变。

## 统一测试结果

本轮统一运行 191 项：189 通过，2 失败。新 GenericMappingAdapter 的 24 项、LingBot 官方 FeatureTransform 行为对照、EpisodeBuilder、Training Contract 与语言完整性回归通过。

失败用例是 `test_group_audit_matches_across_engines_and_preserves_files` 和 `test_joint_check_is_readonly_and_matches_both_audit_engines`。两者均在官方 storage 导入阶段因当前环境未安装 LeRobot 0.4.2 而失败，没有进入实际质量逻辑；不能声称真实官方存储集成已验证。此次未安装额外的大型依赖，也没有运行实际生产数据清洗。

官方存储可独立通过项目 `lerobot` extra 安装，不需要 LingBot 模型配置。新增映射的 V3 audit 接线另有使用受控 storage fixture 的通过用例；该用例不能代替上述真实存储集成验证。
