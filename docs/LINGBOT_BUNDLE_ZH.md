# LingBot training bundle

新增正式包 API `LingBotBundleExporter.export()` 与 CLI：

```bash
lerobot-cleaner export-lingbot-bundle \
  --dataset /data/source \
  --output /data/lingbot-bundle \
  --lingbot-root /repos/lingbot-vla \
  --train-config /configs/my_train_template.yaml \
  --robot-config /configs/my_robot.yaml \
  --config /configs/cleaning.yaml
```

在装有官方 LeRobot、LingBot 依赖、可用 CUDA 的 Linux/WSL Python 环境运行。
模型/processor/tokenizer 的本地资源仍由用户提供；Level 2 不加载 VLA 模型权重。
`--cuda-device` 默认 0，norm 固定单进程避免重复全量统计。

## 配置和映射来源

复用现有 V3 `--config`，包含 semantic_adapter、mapping_config、quality、policy、transforms 等。
默认先调用 clean_v3，再用现有输出检查模式校验；不再次应用原输入的删集/裁剪索引。
v2.1 输入沿用既有 converted_root 与官方转换机制。
`--skip-clean` 只允许已准备好的 v3，复制后仍做输出检查，不修改原数据。

训练模板必填。要求显式提供：
- model：model_path、tokenizer_path；
- data：joints、cameras、norm_type；
- train：chunk_size、max_state_dim、max_action_dim、tokenizer_max_length、resize_imgs_with_padding。

其他训练参数与预处理设置从模板原样保留，不引入另一套训练超参数默认值。
模板中的 ./ 或 ../ 本地模型路径以模板目录为基准；缓存 model ID 保持不变。
导出的 data.train_path、robot_config_root、norm_stats_file 使用最终 bundle 的绝对路径；
data_name 与统一文件名 robot_config.yaml 对齐。移动 bundle 后需要更新这三个绝对路径并重新验证。

robot config 优先复制显式 --robot-config 或 cleaning config 的 robot_config。
否则从现有 adapter 生成 CanonicalFeatureSchema，再序列化已有有序 slice/concat/visual 映射，
用 LingBotRobotConfigSchema 重新验证。state.arm 显式 canonical 名转换成 observation.state.arm；
不从整块 observation.state 向量猜关节名称，无法确定名称时要求提供 robot config 或 named generic mapping。
生成路径额外写 canonical_schema.json，保留解析来源和物理 metadata；不把这些字段塞入官方 YAML grammar。

## 顺序和失败策略

Clean → output quality validation → config export → training compatibility → official norm → Level 2 smoke → ready manifest。
训练兼容性先检查所有 episode，并启用相机采样解码与真实本地 tokenizer 检查。
报告复用现有 training contract，保留 horizon、padding、语言、相机、维度和 joint mask。
官方训练入口当前未转发 train.chunk_size，contract 会阻止 dataset/model horizon 不一致。

norm 由正式包 norm.py 调用官方 train.sh scripts/compute_norm.py，使用官方 VLADataset/FeatureTransform；
cleaner 不计算或仿制统计量。全量运行完成后验证 count、feature keys、shape、有限性和分位数顺序。
只有完整有效 JSON 才从临时文件发布为 norm_stats.json。旧 scripts/run_lingbot_norm.py 保留兼容入口。

Level 2 复用正式包中的 training_batch，验证真实 FeatureTransform、action chunk、图像、语言、
padding/mask 和 collator，不构建大模型。不接受仅 status=passed 而缺少实际验证标志的结果。

映射/维度/horizon/必需相机/语言/tokenizer 错误、norm 失败或不合法、Level 2 失败均阻止 ready。
普通质量 warning、padding 较高不默认阻止；已有 abort/reject policy 仍生效。
`--skip-smoke` 仍计算 norm，但输出 status=unverified、training_ready=false，CLI 返回 2。
成功 ready 返回 0；运行失败返回 1。失败目录保留 status=failed 的 manifest 与诊断，绝不覆盖已有输出。

## 最终目录

```text
output/
├── dataset/
├── robot_config.yaml
├── train_config.yaml
├── norm_stats.json
├── compatibility_report.json
├── cleaning_report/
│   ├── 原有清洗报告
│   └── bundle_validation.json
├── manifest.json
└── canonical_schema.json   # 从 canonical mapping 生成 robot config 时
```

manifest 记录 artifact 相对路径、ready 状态、official_lingbot 后端、LingBot commit/工作区状态、
关键源码 SHA256 与输出配置/norm/report SHA256。源码无 Git 时 commit=null，仍保留关键源码指纹。
规范路径使用最终 output 目录；异常或中断期间该目录不代表 ready，必须读取 manifest。
质量报告和 compatibility_report 分开，bundle validation 不复制 normalization/cleaning 算法。

## 本轮文件

- 新增 v30/lingbot/bundle.py、v30/lingbot/norm.py、training/smoke/batch.py。
- 修改 cli.py、v30/v3.py、v30/pipeline.py、training/smoke/lingbot.py。
- scripts/run_lingbot_norm.py、scripts/lingbot_training_batch.py 改为兼容入口。
- 新增 tests/test_lingbot_bundle.py、本文；更新 README.md。

## 验证范围

统一回归共 330 项：328 通过，2 项已有官方数据访问集成测试因未安装 lerobot 包失败。
本轮 22 项 bundle 测试通过，覆盖编排顺序、失败状态、skip 控制、CLI、路径、保留报告、
canonical YAML 导出、官方脚本命令与 norm 验证。外部执行采用替身；没有实际运行 GPU norm
计算或真实 LingBot Level 2，因此不宣称已验证目标训练环境的端到端导出。
