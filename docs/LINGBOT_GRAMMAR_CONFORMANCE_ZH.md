# LingBot robot YAML grammar：源码审查、实现与一致性验证

## 参考版本与证据

本次检查的是本机官方仓库 `Robbyant/lingbot-vla`，commit
`4eb34b7693a0565c67433f8fac9c59a2e67eb60b`。检查时 `utils.py` 和 `transform.py`
没有工作区修改。`lingbotvla/data/vla_data/utils.py` SHA256：
`527af3d8ad8ada79c89637e315ee36cb7f5c16a685f985a766fae66e03939d84`。

源码：[固定版本 utils.py](https://github.com/Robbyant/lingbot-vla/blob/4eb34b7693a0565c67433f8fac9c59a2e67eb60b/lingbotvla/data/vla_data/utils.py)。
审查覆盖 `check_robot_config`、`get_feature_mapping`、`convert_features`、
`reverse_features`、`apply`、`pad_and_concat`，并搜索了全仓 `convert_from_state` 的使用。
结论限定此版本，不声称已核验网络上的最新实现或未来版本。

## A. 原实现与官方的具体差异

| 语法/行为 | 官方源码行为 | 原 cleaner | 修改后 |
|---|---|---|---|
| states direct string | 支持，保留原 key | 只接受单 target mapping | 支持 |
| images direct string | 支持，保留原 key | 只接受 mapping | 支持 |
| actions direct string | check_robot_config 拒绝 | 拒绝 | 继续明确拒绝 |
| numeric origin_keys 为字符串 | 整个源张量重命名 | 强制 list | 支持，从真实元数据计算维度 |
| origin_keys 列表元素为字符串 | `.items()` 报错 | 拒绝 | 明确拒绝，不能把用户示例当成官方合法语法 |
| origin_keys 列表包含源→start/end 字典 | 顺序切片、最后轴拼接 | 只允许每个字典一个源 | 支持字典内多个源，保持两级顺序 |
| 重复源切片 | 临时加 `*` 区分，再恢复源名 | 已支持简单重复切片 | 保留顺序并与真实输出比较 |
| 负端点/超过维度的端点 | 实际 tensor slicing 按 Python 截断 | 强制 0≤start<end≤width | 按真实 width 规范化，保留原声明端点 |
| null 端点 | 初始化反向映射时端点算术报错 | 拒绝 | 明确拒绝；不是官方可执行语法 |
| subtract_state 省略 | 默认 false | 强制 action 显式写布尔值 | 默认 false |
| convert_from_state | 仅写局部集合，没有后续消费 | 完全忽略 | schema 保留标记；来源严格由 origin_keys 决定 |
| images origin_keys 字符串 | 重命名 | 已支持 video | 同时接受官方 image/video 存储 |
| images origin_keys 切片列表 | 与数值特征使用同一套最后轴拼接 | 不支持 | 表达为派生图像 slices，按 CHW 最后轴（宽）拼接 |
| origin_key 单数 | 没有读取该字段，不产生重命名 | 会因缺 origin_keys 出错 | 明确报告字段错误 |
| target 命名/空 section | 与训练 data_config 成员关系检查；列表可为空 | 强制固定前缀、三节非空 | 不擅自加前缀；可空/省略 section；训练成员验证仍是独立契约 |

因此确有“官方有效但旧 cleaner 报错”的配置，如 direct state/image、字符串 source、
省略 subtract_state、多 source 字典、负切片。原 cleaner 对有 origin_keys 的配置会忽略 convert_from_state；这个固定官方版本同样不据此改变来源，
因此不能把“自动从 state 生成 action”说成原实现缺失的已实现官方功能。原实现缺少的是对该字段
和上游未实现边界的明确表达与诊断。

官方语法例子：

```yaml
states:
  - observation.state.arm
  - observation.state.combined:
      origin_keys:
        - sensor_a: {start: 2, end: 5}
        - sensor_a: {start: 0, end: 2}
        - sensor_b: {start: 1, end: 3}
actions:
  - action.arm:
      origin_keys: action
      # subtract_state 默认为 false
images:
  - observation.images.camera_top
  - observation.images.camera_renamed:
      origin_keys: observation.images.camera_side
```

这些 source 必须存在于官方 dataset feature metadata。target 还须满足实际训练的
`data_config.joints` / `data_config.cameras`；仅 robot YAML 解析成功不等于训练可运行。
特别是 images 的训练 target 必须出现在官方生成的 `observation.images.<camera>` 名单中。

## B. 修改文件

| 文件 | 原因 |
|---|---|
| `lerobot_cleaner/adapters/lingbot_config.py` | 新建有类型的规范化层、维度和来源校验、显式训练前 feature 转换 |
| `lerobot_cleaner/adapters/lingbot.py` | 改为消费规范化 schema；初始化失败释放自有 storage；报告参考版本 |
| `lerobot_cleaner/adapters/schema.py` | canonical 保留 convert_from_state 标记；说明派生图像 slices |
| `lerobot_cleaner/adapters/resolver.py` | 接受派生图像；减 state 的名称替换与官方全字符串 replace 一致 |
| `lerobot_cleaner/v30/planning.py` | 派生图像不能被误当成一个物理相机执行永久 ROI；报错要求指定实际源 |
| `tests/test_lingbot_conformance.py` | 新增真实官方源码一致性、反例及 adapter 边界回归测试 |
| `tests/test_dataset_adapters.py` | 原越界切片反例改为空切片；超过上界本身不是官方错误 |
| `docs/LINGBOT_GRAMMAR_CONFORMANCE_ZH.md` | 记录源码审查、边界、测试命令和结果 |

## C. Schema 与运行边界

```text
robot.yaml
  → LingBotRobotConfigSchema
      NumericFeatureSpec / ImageFeatureSpec / SourceSliceSpec
  → LingBotAdapter
  → FeatureSchema + FeatureSlice
  → FeatureResolver / EpisodeBuilder
  → UnifiedEpisode / readonly TrajectoryView
```

每个数值 target 记录 name、section、顺序 source slices、原声明端点、实际端点、
dimension、direct、subtract_state、convert_from_state。
图像记录 target、direct/rename 的 origin_key，或派生图像的顺序 slices。
派生图像以 `camera_column=None, slices=(...)` 表达，避免错误地指向单个视频。
它是训练特征映射，不会自动把源视频裁剪或永久拼接成新视频；永久修改仍须显式 TransformPlan。

canonical 数值视图继续提取原始绝对量，避免清洗阶段偷偷把数据变成 delta。
需要比较官方训练前结果时，显式调用
`adapter.robot_schema.convert_features(item, subtract_state=True)`。
该接口返回 feature 张量的 NumPy 视图结果副本，不修改输入；不包含 task、padding mask、
normalization、resize、tokenizer 或 model padding。这些仍属于训练 preprocessing。

两个顺序不能混为一谈：
- `states/actions/images` target 列表和每个 target 内的 slices 都按 YAML 顺序。
- 官方 `pad_and_concat` 最终按 `data_config.joints` 排序并补齐维度；相机按 training cameras。
  robot YAML 单独不能决定整个模型 tensor 的最后顺序。测试直接调用官方此函数验证了这一点。

## D. convert_from_state 的真实行为

官方 `get_feature_mapping` 会 pop 这个字段，并把 true target 加入局部
`actions_convert_from_state` 集合。该集合没有保存为实例属性，也没有被后续方法读取。
`convert_features` 只根据 `origin_keys` 取数据，`apply` 只额外执行 subtract_state 等处理。
因此，此版本中该字段**不执行**自动选 state、下一帧移位、切片替换或维度推断。

- `origin_keys: action` + true：仍读取 action。
- `origin_keys: observation.state` + true：读取该原始 state 列，原因是 origin_keys。
- 切片列表 + true：使用该列表的顺序、实际切片和维度。
- true 但没有 origin_keys：官方登记 target，却不产生该 action tensor。cleaner 报明确错误，
  不自行猜测相应 state，也不伪造未来 state action。
- 是否减 state 仍由 subtract_state 单独控制，使用命名配对后的规范化 state target。

cleaner 同时保存 marker 和实际 source slices；测试覆盖来源 state/action 两种情况，
包括没有对应 state target、但有合法显式原始 state source 的配置。

## 明确的检查边界

不复制官方偶发的错误、静默忽略或不完整结果：未知字段、单数 origin_key、空切片、重复 target、
显式重复 YAML key、错误端点类型、source 不存在、subtract_state 配对缺失/维度不同均提前报错。
即使 torch 某些维度可广播，cleaner 仍要求减 state 的 feature 维度相等，避免意外广播。
官方能产生零维空 feature，cleaner 的 canonical 数据模型不接受零维 feature。
这些是明确的严格检查，不声称所有能让官方构造函数返回的输入都是可用配置。

YAML merge 的明确覆盖保留 SafeLoader 行为；共享会被官方修改的 feature mapping alias
会导致上游后续 target 消失或标记变化，因此 cleaner 明确拒绝，要求写成独立 mapping。
正常复用标量、源切片列表和不共享可变外层 mapping 的 merge 不受影响。

超界/负端点的前向 tensor 结果与官方相同；官方 reverse_features 用声明端点计算反向偏移，
并不保证这种配置可逆。本次验收是前向 feature mapping，不宣称实现所有反向 reconstruction。

## E. 新增测试

`test_official_mapping_and_tensor_conformance`：15 个参数化 case，覆盖 A–H，另包含
多源字典顺序、默认 subtract_state、负/超界端点、派生图像切片、target 顺序。
比较官方 target 列表、来源、声明/实际 slices、维度、canonical 提取、convert_from_state、
图像映射，以及 `apply(do_nomalize=False)` 的数值结果；检查 cleaner 不修改原数组。

其余测试覆盖：
- `test_convert_flag_without_origin_is_an_upstream_noop`：证明官方不自动生成 action。
- `test_official_rejects_apparent_shorthands`：4 种看似合理却被官方拒绝的写法。
- `test_singular_origin_key_is_not_a_rename`：官方单数字段不产生输出。
- `test_yaml_order_is_distinct_from_training_padding_order`：真实模型拼接顺序边界。
- `test_contextual_invalid_or_ambiguous_config`：12 种包含上下文的明确报错。
- `test_reject_duplicate_yaml_keys`、`test_image_storage_and_empty_sections`。
- `test_existing_droid_yaml_preserves_slices`：现有 DROID 7+1 维映射保持。
- `test_episode_uses_explicit_action_source_without_applying_delta`：2 个 flag case，验证 readonly episode。
- `test_yaml_merge_override_matches_safe_loader`、`test_shared_mutating_yaml_mapping_is_rejected`。
- `test_other_adapters_keep_their_canonical_contract`：LeRobot / GR00T 语义边界保持。

测试加载 checkout 中未修改的 `utils.py` 及真实 `transform.py`，没有复制/模拟官方映射算法，
也没有 mock torch。只隔离包初始化，避免引入无关的大模型训练 import。
通过 `LINGBOT_ROOT` 可选择另一个官方 checkout；找不到源码或必要依赖时会明确 skip，
这不能算作官方一致性通过。

## F. 运行结果

最终相关测试：**68 passed，0 failed，0 skipped**。
其中新增文件 42 项，EpisodeBuilder 20 项，现有 LingBot 配置 6 项。

```powershell
$env:LINGBOT_ROOT = "<官方 lingbot-vla checkout>"
$env:PYTHONPATH = "$PWD/.verification-deps"
python -m pytest tests/test_lingbot_conformance.py tests/test_episode_builder.py tests/v30/test_lingbot_configs.py -q --tb=short -p no:cacheprovider --basetemp outputs/lingbot-conformance-new-run
```

basetemp 使用新目录，避免覆盖历史测试产物。`einops` 安装在 Git 忽略的 `.verification-deps`，
未改项目运行依赖；torch 使用本机已有版本。

过程中：最初缺 einops、系统临时目录权限不足属于环境问题，安装隔离依赖并使用项目临时目录后解决。
一次官方对照失败揭示 null slice 初始化报错，已收紧实现并改成负面一致性测试。
没有通过改官方源码、伪造依赖或跳过该失败来取得最终通过结果。

未执行整个 pytest 套件，也未运行真实 DROID/LIBERO 数据清洗、v2.1 转换、视频重写或模型训练。
这里的回归验证覆盖 adapter / assembly / 配置边界，不等于这些完整数据管线已经验证通过。
