# configs/ 目录说明

本目录按"谁来消费这份配置"分组。**手写源** 与 **生成产物** 分开维护，
生成产物不要手改——修改 `embodiments/` 下的 spec 后用命令重新生成。

## cleaning/ — v3 清洗引擎配置（手写）

`audit-v3` / `clean-v3` 通过 `--config` 传入的清洗配置（`V3Config` schema）。

新版检查、policy、TransformPlan 和变换配置见 [规则说明](../docs/V3_RULES_AND_TRANSFORMS_ZH.md)。

| 文件 | 用途 |
| --- | --- |
| `training_contract.example.yaml` | 独立 LingBot training contract、padding 与训练兼容性诊断 |
| `v3_quality_policy.example.yaml` | 新版质量规则、判断策略和显式变换的完整示例 |
| `droid_v3.yaml` | DROID 数据默认清洗配置：分批 streaming 引擎、保守策略（不删帧、非有限值报错） |
| `droid_v3_referenced.yaml` | 保留旧文件名，现已改为官方 strict 读取；残留分片须先单独修复，不再自动排除 |
| `libero_v3.yaml` | LIBERO 数据走通用 v3 引擎时的清洗配置，见 [docs/LIBERO_V3.md](../docs/LIBERO_V3.md) |

统一入口支持 `semantic_adapter: auto/lingbot/lerobot/groot`。v2.1 输入需设置 `converted_root`，官方转换副本后按 v3.0 读取。参考 `cleaning/groot_v21_official.yaml`。

## profiles/ — 数据集审查 profile（手写）

review 层（`dataset/episode_review.py` + `review_profile.py`）使用的数据集契约
（特征名/维度/相机清单）与质量启发式阈值。`scripts/run_libero_clean.py --profile`
指定，默认 `libero_fastwam.yaml`。profile 决定"审查什么、阈值多少"，
与 `cleaning/`（决定"怎么清洗"）相互独立。

## embodiments/ — embodiment spec 源文件（手写，勿放生成物）

`lerobot-cleaner generate-lingbot-config <dataset> --spec configs/embodiments/<name>.yaml`
的输入：单份 YAML 声明某本体的关节切片、相机映射与训练起始参数。

- `droid_franka.yaml`：已对照真实 DROID v3.0 数据核验（7 关节 + 1 连续夹爪 + 3 相机）。
- `r1pro.yaml`：**未验证模板**，非连续切片，使用前必须对照真实数据核对。

## robot_configs/ 与 vla/ — 生成产物（勿手改）

`generate-lingbot-config` 的输出对（`--robot-config` / `--train-config` 的默认落点）：

- `robot_configs/<name>.yaml`：LingBot 机器人特征映射。
- `vla/<name>.yaml`：LingBot 训练起始配置；其中 `robot_config_root` 指向本目录的
  `robot_configs/`，`data_name` 必须与 robot config 文件名一致。

两者在发布前都会经过 `lerobot_cleaner/v30/lingbot/config.py` 的 `validate_mapping`
静态校验。要改映射请改 `embodiments/` 后重新生成（命令默认拒绝覆盖已有文件）。

## 旧目录：presets/

项目根的 `presets/` 属于 GR00T v2.1 时代（`run` 命令的 R1Pro 清洗预设），
与 v3 流程无关，保留用于旧格式数据。

通用 LeRobot feature mapping 示例见 [mappings/generic.example.yaml](mappings/generic.example.yaml)，在清洗配置中设置 `semantic_adapter: generic` 和 `mapping_config`。

Quality groups 推荐 `feature` / `features`。DROID 配置已迁移；LIBERO 新增 [语义配置](cleaning/libero_v3_semantic.yaml) 与 [布局 mapping](mappings/libero.example.yaml)，原配置保持兼容。详见 [说明](../docs/SEMANTIC_QUALITY_GROUPS_ZH.md)。
