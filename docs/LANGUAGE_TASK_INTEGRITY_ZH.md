# Language / Task Integrity

## 修改前的数据路径与缺口

1. `CanonicalFeatureSchema` 原先只有 states、actions、cameras；`EpisodeRef.tasks` 是未纳入质量检查的旧接口，V3 builder 创建引用时将其初始化为空列表。
2. v2.1 旧读取器从 `meta/tasks.jsonl`、`meta/episodes.jsonl` 和样本 `task_index` 读取任务；推荐 V3 路径先按既有配置经官方转换，再使用 `LeRobotDataset.meta.tasks`、官方 episode metadata 和样本列。
3. DROID / LIBERO 的原始 `task_index`、任务表和相关字符串列没有被统一适配器主动删除。结构修改后的官方 writer 根据任务文本重新建立 task ID。原始信息保留不等于已有语言完整性保障。
4. `LingBotAdapter` 继承官方 storage，因此能访问任务表，但此前没有统一的 canonical episode/sample 语言接口。
5. 原有 dataset review 有任务频率、非空文本校验；Training Contract 有 LingBot 专属 task lookup 和缺失统计。通用 `dataset_quality.language`、来源冲突检查、编码检查和真实分词截断报告尚不存在。
6. 无法解析的索引、缺失任务、episode/sample 映射冲突可能未被统一检查；部分旧路径会直接抛错，中断报告。V3 的空 `EpisodeRef.tasks` 也容易被误认为任务已丢失。

## 当前边界

```text
LeRobotDataset / 已有 legacy loader
  → Adapters + FeatureResolver
  → CanonicalFeatureSchema.language
  → EpisodeBuilder → UnifiedEpisode.language (EpisodeTask / TaskInfo)
  → Language / Task Integrity → QualityReport + policy → TransformPlan

LingBot 严格官方 task lookup → 同一通用 TaskInfo 类型
  → 可选真实 processor.tokenizer → training_readiness.tokenization
```

`LanguageFeature` 声明索引列和文本候选列；默认是 `task_index` 以及 `task`、`language`、`instruction`、`prompt`。适配器可覆盖 `get_language_feature()`，不需要依赖模型。`TaskInfo` 包含原始索引、原文、来源、episode ID 和可选原始样本位置。`EpisodeTask` 同时保留 canonical 样本和其他来源证据。

解析优先级为唯一任务表映射、样本字段、明确且不冲突的 episode 声明；无 episode 声明时可读取 dataset metadata 的任务字段。不会 strip、替换文本、填充缺失 ID，或以有效备选值掩盖无效的权威值。无效表映射即使存在可用样本文本，也保留完整性错误。

`UnifiedEpisode.language` 根据当前行只读解析；统一 frame filter 后保留源样本位置。异步语言流按 EpisodeBuilder 的同一时间规则对齐；真实缺失样本仍报错，人工 padding 不计为缺失语言。数值 `TrajectoryView` 保持原有只读数组职责。

## 通用检查与报告

`core/quality/integrity/language.py` 不依赖 LingBot、transformers、Qwen 或 tokenizer，检查：

- 索引缺失、非整数、负数、超出官方声明范围、无法映射、重复表索引及表计数不一致。
- 每条 episode 和每个真实 sample 的任务覆盖；默认不允许 episode 内指令变化。
- None / NaN、空串、纯空白、非字符串、无法编码为 UTF-8 的文本。
- 同一样本不同字段、任务表、episode 声明之间的冲突。
- 文本长度、占位符、数字和控制字符提示，以及跨 episode / sample 的重复统计。

仅完整性错误使 `language_integrity.passed=false`。重复统计为 INFO；短文本、长文本、占位符等为 WARNING，不使检查失败，即使 policy 配置为 reject 也不会仅凭这些提示删除 episode。检查器不会修改源数据，policy 只生成计划。

V3 `audit-v3`、dry-run 和实际 clean 的输出复核均包含 `dataset_quality.language`：

```yaml
episodes_total: ...
episodes_evaluated: ...
episodes_with_task: ...
episodes_without_task: ...
episodes_with_incomplete_task: ...
episodes_failed: ...
missing_ratio: ...
task_missing_ratio: ...
unique_task_count: ...
duplicate_task_count: ...
invalid_task_index_count: ...
empty_task_count: ...
inconsistent_mapping_count: ...
task_frequency_distribution: ...
sample_task_frequency_distribution: ...
```

`episodes_with_task` 表示至少一个样本有完整字符串，部分缺失另计 `episodes_with_incomplete_task`。缺失率按没有任何有效文本的已检查 episode 计算；禁用时状态为 `disabled`、比率为 null，不声称检查通过。

`task_frequency_distribution` 按“包含该文本的 episode 数”计数；`duplicate_task_count` 为各文本该计数减一后的非负总和。另有 sample 频率，避免长 episode 改变 episode 重复率。索引错误按样本计数，任务表错误可能在各 episode 重复出现；无效文本按 episode 内不同来源/值计数，不等同于坏帧总数。

各 episode 的 `checks.language_integrity.metrics` 保存错误、提示和 provenance。无法编码的文本仅在报告中转义，canonical 和原始数据保持原值。清洗报告另有 `language_input` / `language_output`；review bundle 输出 `language_quality.json`。输入和输出 episode ID 的关联沿用 `episode_map`，任务 ID 可由官方 writer 重新编号。

任务表映射问题不再由 cleaner 的 storage 校验提前抛错，旧 profile review 也不再抢先中断任务报告。官方库自身不能打开的损坏数据，以及不相关的 storage/profile 错误，仍然可能阻止扫描；本层不伪造这类数据的完整报告。writer 仍拒绝无效 ID、无效任务文本或冲突的样本 task，不会通过整数强转暗中修复语言映射。

## 配置

将下列段合并到现有 V3 配置：

```yaml
quality:
  language:
    enabled: true
    allow_instruction_changes: false
    min_length: 2          # null 关闭短文本提示
    max_length: 4096      # null 关闭长文本提示
    check_control_characters: true
    check_numeric: true
    placeholders: ["n/a", "unknown", "task"]  # [] 关闭
policy:
  rules:
    language_integrity:
      on_fail: warn       # 可显式配置 report / reject_episode / abort
```

允许 episode 内指令变化时只关闭跨样本唯一性限制，不豁免同一样本的来源冲突。任务表不存在且数据只提供 episode 文本时，不强制凭空增加 `task_index`；LeRobot 格式本身所需的索引字段仍受其结构检查约束。

## LingBot 官方对齐与真实分词

对照本地官方 checkout，版本 `4eb34b7693a0565c67433f8fac9c59a2e67eb60b`：

- `lingbotvla/data/vla_data/base_dataset.py`：`meta.tasks.iloc[task_index].name` 为实际文本，不能换成便利的其他文本列。
- `lingbotvla/data/vla_data/utils.py`：拼接后 prompt 为 `[item["task"]]`。
- `lingbotvla/data/vla_data/transform.py::prepare_language`：仅在缺失时补 `<bos>` 和末尾换行；调用 tokenizer 时 `padding="max_length"`、`padding_side="right"`、`truncation=True`、`return_tensors="pt"`。
- `tasks/vla/train_lingbotvla.py` 与 `lingbotvla/models/auto.py`：从 `model.tokenizer_path` 加载 AutoProcessor，使用 `processor.tokenizer`。
- `max_length` 来自既有 Training Contract 的 `train.tokenizer_max_length`，缺省为官方 48，不硬编码其他模型的长度。

`training/compatibility/tokenization.py` 用同样的提示格式和实参做实际分词，再对同一提示做不截断分词获取原始 token 数。检查有效 attention mask、非空 token、整数 ID、右侧 padding、固定宽度及前后 token 数的一致性。无有效 token 或 tokenizer 异常为 Training Compatibility ERROR；截断为 WARNING。

每条去重文本包含 `token_count`（格式化提示的不截断有效 token 数）、`effective_token_count`、`max_length`、`truncated`、`truncation_ratio`（丢弃有效 token / 原始有效 token）。汇总包含截断文本比例和按样本频率加权的截断比例。不会靠字符长度估算 token。

```yaml
training_check:
  target: lingbot
  robot_config: ../robot_configs/droid_franka.yaml
  train_config: ../vla/droid_franka.yaml
  tokenization:
    enabled: true
    tokenizer_path: null  # 默认使用训练 YAML 的 model.tokenizer_path
```

也可使用独立 `check-training` 命令的 `--tokenize` 与可选 `--tokenizer-path`。本地目录建议使用绝对路径，也支持本机已缓存的模型 ID；相对 tokenizer 路径遵循训练入口的工作目录语义。只读取本地资源，不自动下载。需要匹配的 processor/tokenizer、transformers 与 torch，不加载模型权重。

默认不启用真实 tokenizer 检查，以保持原静态 Training Contract 的轻量入口；此时明确报告 `not_run`，不能视为分词兼容性通过。启用后资源不可用报告 `unavailable` 和 ERROR。此检查只验证语言分词，不替代完整 preprocessing/collator smoke 或模型 forward。

## 本次验证范围

新增 `tests/test_language_integrity.py`，覆盖任务来源、映射、错误文本、编码、重复统计、指令变化、删帧关联、异步对齐、人工 padding、分词参数和截断统计等回归场景。依据本项目“只修改、不运行”的要求，本次只做源码审阅，没有运行项目程序、测试、编译、lint、转换或实际 tokenizer；不宣称新增用例已通过。
