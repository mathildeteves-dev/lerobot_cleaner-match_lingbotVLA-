# 这份 DROID 数据如何清洗并接入 LingBot-VLA

更新：默认 DROID 配置已切换为分批引擎。服务器完整数据的操作和阶段级续跑说明见
[服务器分批清洗步骤](STREAMING_SERVER_ZH.md)。下文“100 集、19 集失败”等检查结论仅对应本地样例，不能套用到完整 DROID。

## 1. 先确认：你给的数据是什么

本次逐行读取了 `数据实例/droid_100_lerobotv3/droid_100_lerobotv3` 的数据表与元数据。
版本来自 `meta/info.json`，不是仅根据文件夹名字判断。

| 项目 | 实际检查结果 |
| --- | --- |
| 格式 | LeRobot v3.0，不需要再升级格式 |
| 机器人 | `robot_type: Franka`；单臂，而非 R1Pro 双臂 |
| 数据规模 | 100 集、32,212 帧、47 个任务、15 Hz |
| 状态/动作 | 各 8 维：前 7 维关节位置，最后 1 维夹爪位置 |
| 相机 | exterior_1_left、exterior_2_left、wrist_left |
| 存储 | 全部演示合并在一个数据 parquet；每路相机一个共享视频文件 |
| 数值 | 所有声明为数值的字段均未发现 NaN / Inf |
| 演示结果 | 19 集标记为失败；未据此自动删除 |
| 演示长度 | 最短 13 帧，最长 1,627 帧；未自动过滤短演示 |

已逐行核对：`action[:7]` 与 `action.joint_position` 完全相等，
`action[7]` 与 `action.gripper_position` 完全相等，状态也存在同样的重复存储。
`action` 的前 7 维不等于 `action.joint_velocity`。因此这里选择位置字段，不能当作速度输入。
字段名与数值一致性仍不能单独证明真机控制单位、控制周期和指令执行语义。

夹爪值是 [0, 1] 连续值，不应未经确认就变成 0/1。
源目录 `images/` 中有 6 个零字节占位文件，不是元数据声明的训练视频；
输出不复制它们，但没有删除原文件。

## 2. 为什么修改这些文件

| 文件 | 用途与修改理由 |
| --- | --- |
| `lerobot_cleaner/v30/v3.py` | 新增原生 v3 检查/清洗，按逐集偏移处理共享 parquet 与视频。旧 v2.1 清洗器依赖 modality.json 和逐集视频，不能直接套用。 |
| `configs/cleaning/droid_v3.yaml` | 决定如何清洗；默认只检查、保持数值，并声明重复字段之间的对应关系。 |
| `configs/robot_configs/droid_franka.yaml` | 决定训练时如何读取原始字段：7 关节、1 夹爪和 3 相机。它不是格式转换配置。 |
| `configs/vla/droid_franka.yaml` | LingBot 训练起始配置，包括机器人名、相机列表、归一化路径、模型与训练参数。硬件相关参数仍需实际试跑。 |
| `lerobot_cleaner/v30/lingbot/config.py` | 检查切片不越界、相机存在、映射宽度不超过容量，以及动作减状态时两者维数一致。 |
| `scripts/run_droid_clean.py` | 保存可复用的清洗入口；默认路径按脚本位置解析。 |
| `scripts/run_lingbot_norm.py` | 到真正的 LingBot 源码目录启动其 compute_norm.py，并传入绝对路径，避免在清洗器目录找不到 train.sh。 |
| `lerobot_cleaner/v30/lingbot/vla.py` | 仅处理 v2.1→v3.0；转换不再强制需要机器人 YAML，保留清洗报告，拒绝重复转换 v3。 |
| `tests/test_v3.py` | 测试不改原始数据、共享视频偏移、NaN 策略、重复字段同步、配置越界与命令行等。 |
| `tests/test_lingbot_vla.py` | 修复误贴在文件末尾的执行代码，并补充转换失败保护/报告保留测试。 |
| `pyproject.toml` | 增加可选视频解码依赖与更新仓库地址；已有 lingbot 额外依赖保留供旧格式升级使用。 |

原有 R1Pro 配置保留并标为未验证模板。重复的 `scripts/scripts/` 启动文件已移至
`scripts/run_compute_norm_r1pro.sh`；它仅用于未来实际符合该映射的 R1Pro 数据。

## 3. 这次清洗具体做了什么

默认配置不会制造并不存在的“脏数据”，也不会为了看起来清洗过而删除数据。

- 检查帧编号、集编号、任务索引、时间戳、特征维数、视频文件和逐集时间区间。
- 检查所有数值列的有限性；默认遇到 NaN/Inf 停止，不自行填补。
- 保留所有行、失败标记、文字任务、外参、额外字段，以及视频索引和偏移。
- 重新计算数值字段的全局和逐集统计；视频未变，保留原视频统计。
- 复制声明的视频，核对原文件与副本的 SHA256；不转码。
- 先写临时目录并检查，通过后才发布新输出目录。已有目录不会覆盖。

本次真实数据运行的结果是 **改动数值 0，删除行数 0**。
这表示输入通过了当前检查，不表示全部演示都适合训练。
失败演示是否保留、是否剔除短演示或静止段，属于你需要确定的训练数据策略。
当前 v3 实现不支持删帧、删集、重采样或图像裁剪；不要把 v2.1 的 R1–R7 配置传给它。

如果以后确实有需要，可在清洗 YAML 中把 `nonfinite` 改成 `interpolate`：
仅在同一集内部逐维插值，边缘用该集最近的有效值；整集某一维全无效时仍报错。
`bounds` 默认空，可为明确知道物理范围的浮点字段设置上下限，切勿猜测关节限位。
对位置做插值或限幅后，额外的速度/笛卡尔字段不会自动重算运动学；本次没有启用这些修改。
配置采用严格字段校验，不支持的选项会报错，不会悄悄忽略。

旧 `memory` 引擎仍会全量读取，默认限制 100 万帧；不传配置的通用 CLI/API 保留此行为。
当前 DROID 配置已切换到 `streaming`：每批 8,192 行，最多缓冲 100,000 帧的一段演示，
数据集总规模保护设为 3,000 万帧。全局分位数默认从最多 32,768 行/字段的均匀抽样估计，
全局 mean/std/min/max/count 累计全部帧，逐集分位数仍按该集完整数据计算。
完整数据能否通过所有结构/数值检查，需在服务器实际运行后确认。

## 4. 在你的 Windows 电脑上运行

打开本项目根目录的终端，确认能看到 `pyproject.toml`。

```powershell
# 初次安装到当前 Python 环境；不需要为当前 v3 数据安装完整 LeRobot 训练依赖。
python -m pip install -e ".[dev]"

# 只读检查，不生成新数据
python -m scripts.run_droid_clean --audit-only

# 生成新副本；输出已存在时会拒绝覆盖
python -m scripts.run_droid_clean

# 想另存一份，用不同输出目录
python -m scripts.run_droid_clean --output "../清洗结果/droid_100_clean_v3_second"

# 回归测试
python -m pytest -q
python -m ruff check .
```

默认输入位于本项目同级 `数据实例/droid_100_lerobotv3/droid_100_lerobotv3`，
默认输出位于同级 `清洗结果/droid_100_clean_v3`。
移动项目或更换样本时可以传入 `--dataset "实际数据根目录"`。
数据根目录应直接含 `meta`、`data`、`videos`，不是它外面的同名包装文件夹。

输出 `cleaning_report/` 中包含：

- `report.md` / `report.json`：清洗结果、数值改动、视频哈希。
- `cleaning_config.used.yaml`：本次真正使用的配置。
- `input_audit.json`：全部输入检查结果（使用 DROID 启动脚本时生成）。
- `lingbot_mapping_validation.json`：实际数据与映射配置的静态检查结果。

若需要逐帧解码验证，请先安装可解码 AV1 的 PyAV，再执行：

```powershell
python -m pip install -e ".[v3-video]"
python -m scripts.run_droid_clean --audit-only --verify-videos
```

本次环境没有 PyAV / ffmpeg；安装 PyAV 的尝试被网络代理连接故障阻止。
因此现有报告里的 `metadata_only` 只说明索引、文件存在性、时间区间等通过，
**不代表视频像素已逐帧解码验证**。哈希相同只能证明复制无损。

## 5. 为什么 v3 还需要映射

v3 规定文件如何存储；机器人映射规定“8 个数字分别喂给模型的哪个位置”。
本次实际读取的维度是 7+1。训练配置保留 `arm.position: 14`、`effector.position: 2`
的容量供 LingBot 填充，剩余部分是占位，不代表数据里还有另一只机械臂。

`subtract_state: false` 明确选择保留原数据的位置动作，不转成相对当前状态的差值。
夹爪（effector.position）和末端位置（end.position）不允许启用 subtract_state，静态校验会拒绝。
只有支持差分的关节位置可以按需要启用相对动作。

如果你决定按所用预训练方案改成相对动作，必须同步确认部署端如何还原，
并用改后的映射重新计算归一化统计；不要只改 YAML 就直接控制机器人。
相机名称忠实区分两个外部视角和一个腕部视角，未把第二外部视角伪装成右腕。

## 6. LingBot 归一化与训练准备

这一阶段请在具有相应 GPU、模型权重与依赖的 **Linux / WSL LingBot 环境**完成。
清洗器的 Windows 环境不等于 LingBot 训练环境。
安装 `[lingbot]` 只提供旧数据转换所用的 LeRobot 版本，**不会安装整套 LingBot-VLA**。
请按 [LingBot 官方仓库](https://github.com/Robbyant/lingbot-vla) 准备训练环境。

将清洗结果放到训练机器可访问的目录；在同一环境中安装本清洗器，
然后从本清洗器项目根目录运行以下示例，把路径换成该机器上的路径：

```bash
# 先只显示将运行的命令，不启动 GPU 计算
python -m scripts.run_lingbot_norm \
  --lingbot-root /absolute/path/to/lingbot-vla \
  --dataset /absolute/path/to/droid_100_clean_v3 \
  --dry-run

# 确认之后正式运行
python -m scripts.run_lingbot_norm \
  --lingbot-root /absolute/path/to/lingbot-vla \
  --dataset /absolute/path/to/droid_100_clean_v3
```

脚本使用 LingBot 自己的 `train.sh` + `scripts/compute_norm.py`，
显式传入绝对的训练配置路径、机器人映射目录、数据目录与统计输出路径。
默认输出 LingBot 目录下的 `assets/norm_stats/droid_franka.json`，已存在则拒绝覆盖。
`--cuda-devices` 默认 0，只接受一个 GPU 序号或 GPU/MIG UUID，不接受 0,1 等列表。
归一化固定单节点单进程，并使用本机 rendezvous 地址，不继承外部多节点设置。
启动时为 Bash 启用 pipefail，使计算失败能穿过 tee 传递给启动器。统计先写入目标目录下的临时子目录；
只有进程成功，且 JSON 的映射字段、形状、有限性、标准差、分位数顺序和样本数检查通过后，才发布目标文件。
样本数必须等于完整输入数据的帧数；绝对动作统计为一维，差分关节动作统计按动作块检查二维形状。
发布使用同一文件系统的硬链接以避免覆盖并发出现的文件，因此输出文件系统需支持硬链接；不支持时会报错且不发布结果。
该脚本不是训练脚本，计算统计尚未在本机真实 LingBot 环境中执行。

训练时使用 `configs/vla/droid_franka.yaml`，但必须把其中以下路径改为训练机器上的值，
或按照 LingBot 的启动方式通过命令行覆盖：

- `data.robot_config_root`：本清洗器 `configs/robot_configs` 的绝对路径。
- `data.train_path`：清洗输出的绝对路径。
- `data.norm_stats_file`：上一步真正生成的 LingBot 归一化 JSON。
- `model.model_path` / `model.tokenizer_path`：实际可用模型与分词器位置。
- `train.output_dir`：训练输出位置；其余批大小、并行方式、步数按 GPU 与实验设置。

`meta/stats.json` 是数据集统计，不是映射后的 LingBot 归一化文件，不能直接拿来替代。
建议先完成“加载一个 batch、检查 state/action/image 形状、运行一次前向/反向”的冒烟测试，
尤其检查最短演示的动作块填充，再启动长时间训练。
当前训练配置是基于数据结构的起点，不承诺在未知 GPU、依赖与权重组合上直接跑通。

## 7. 发布到 GitHub 前

本次没有提交或推送 GitHub，也没有上传原始数据。
只提交源代码、配置、测试、说明文档；不要提交视频、parquet、模型权重或个人绝对路径报告。
先查看 `git diff` 与 `git status`，保留上游作者和 Apache-2.0 许可证信息，
在 README 说明新增能力与尚未验证的范围，再按你的正常提交/推送流程发布。

## 8. 验证边界

已完成：真实数据结构检查、全部数值扫描、DROID 映射静态检查、保守清洗副本生成，
以及输入/输出内容和视频哈希核对。
更新后的回归测试：72 项通过、6 项因缺少 ffmpeg 跳过；Ruff 检查通过（包括分批处理测试）。
测试覆盖的是本项目逻辑；转换测试模拟上游转换器，不等于已实际执行官方转换。

尚未完成：视频全解码、官方 LeRobot 运行时加载、LingBot 归一化、训练冒烟测试与真机控制。
“数值有限 / 文件结构正确 / 静态映射通过”均不能替代这些验证，也不代表动作安全或任务成功。
