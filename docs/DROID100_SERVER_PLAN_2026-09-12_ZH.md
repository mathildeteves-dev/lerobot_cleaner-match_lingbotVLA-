# DROID 100 集数据：兼容性检查与服务器方案

日期：2026-09-12。对象：本次提供的 droid_100_lerobotv3 目录；不是历史服务器的 95,658 集数据。

## 检查结果

- 根目录直接包含 meta、data、videos，不需要再向下进入同名目录。
- LeRobot v3.0，Franka 单臂；100 集、32,212 帧、47 个任务、15 Hz。
- observation.state 和 action 均为 8 维：7 个关节位置、1 个夹爪位置。
- 三路相机：exterior_1_left、exterior_2_left、wrist_left；180×320、AV1。
- 1 个数据 parquet、3 个视频；data 11,779,676 字节，meta 1,484,777 字节，videos 474,763,670 字节，合计约 465.42 MiB。
- strict 文件选择通过，无额外未引用数据分片。
- 全量数值/索引/视频区间审计、四组别名关系、LingBot 静态映射均通过；未发现 NaN/Inf。
- 19 集标记为失败；默认配置保留这些演示。
- 本地未安装 PyAV，视频像素的完整解码需要在服务器完成。

使用 configs/cleaning/droid_v3.yaml 和 configs/robot_configs/droid_franka.yaml。
不使用旧版 run，不执行 export-lingbot，不使用 droid_v3_referenced.yaml。
以上结论只适用于本次扫描的文件；服务器数据不同时需要重新审计。

本次机器可读检查记录位于 outputs/droid100_compatibility/。
clean_validation.json 记录真实清洗副本的验证结果；cleaned_v3/ 是本地验证副本。

## 资源计划

以下是工程建议，未测量服务器峰值内存或训练显存；不是硬性最低配置。

| 阶段 | 建议资源 | GPU |
|---|---|---|
| 这份 100 集数据的审计、清洗、视频解码 | 4–8 vCPU，8 GiB RAM 起步，16 GiB 更宽裕；源数据上传后另留至少 10 GiB 可用磁盘，系统/环境盘另计 | 0 张 |
| LingBot 专用归一化 | 4–8 vCPU，16 GiB RAM，复用已有 LingBot 环境 | 当前启动器要求单 GPU；可先复用一张 16–24 GiB 卡，不需要为统计专门租 A800 |
| 后续 4B 模型训练冒烟测试（可选，未实测） | 先按实际模型和精度配置评估；建议 64 GiB 系统 RAM、至少 100 GiB 可用模型/缓存/检查点空间并持续监控 | 1 张 A100/A800 80 GiB 可作为试跑起点，不保证当前 YAML 一定能装下 |

清洗与 PyAV 解码走 CPU，GPU 显存不会替代系统 RAM。默认磁盘预检包括 5 GiB 保留量，因此不能只预留一个 0.5 GB 副本的空间。
正式训练若显存不足，需要调整冻结范围、精度、激活检查点或多卡分片后重估；不建议直接按本方案预订长时间训练。
这里只确认清洗能力，归一化/训练依然需要真实 LingBot 环境验证。

## 服务器执行

假设项目和数据分别上传至以下位置；实际目录不同，只修改变量。
DATASET 必须指向直接包含 meta/info.json 的目录。请上传当前修复后的整个项目，新增文件尚未全部进入 Git 提交，不能仅靠旧远端克隆获得全部修复。

```bash
set -euo pipefail
PROJECT=/code/lerobot_cleaner-match_lingbotVLA_3.0
DATASET=/data/droid_100_lerobotv3
OUTPUT=/data/droid_100_clean_v3
CONFIG="$PROJECT/configs/cleaning/droid_v3.yaml"
cd "$PROJECT"

test -f "$DATASET/meta/info.json"
df -h /data
free -h
# 容器还需核对平台/cgroup 实际内存限额；free 可能显示宿主机内存。

# 先激活已有清洗环境；没有可用环境时再执行以下两行：
# python3 -m venv .venv
# source .venv/bin/activate
python -m pip install -e ".[v3-video]"

mkdir -p logs
# 第一步：只读数值/结构检查
python -m scripts.run_droid_clean \
  --dataset "$DATASET" --config "$CONFIG" --audit-only \
  2>&1 | tee logs/droid100_audit.log

# 第二步：正式清洗、复制哈希校验和输出视频完整解码
python -m scripts.run_droid_clean \
  --dataset "$DATASET" --config "$CONFIG" --output "$OUTPUT" \
  --verify-videos \
  2>&1 | tee logs/droid100_clean.log

cat "$OUTPUT/cleaning_report/report.md"
```

第二步使用 PyAV 解码 AV1；若缺少解码器应修复环境再运行，不能把 metadata_only 当作视频通过。
长任务可放到 tmux 并保留日志。若中断后出现 OUTPUT.partial，先确认旧进程已结束；同输入、同配置可在原命令上加 --resume。
如果留下 .lock 文件，仅在确认旧进程已结束后处理。完成的 OUTPUT 不支持覆盖，不要对完成输出重复运行。
当前恢复是阶段级；未完成的数值阶段和视频审计可能重跑。

## 验收标准

- episodes=100，frames=32212，rows_removed=0。
- 当前数值配置下 changed_values 预期为空；没有设定限幅，且本次扫描未发现非有限值。
- selected_files 为 1 个，无 excluded_files。
- video_sha256 有 3 项，video_verification 为 full_decode。
- lingbot_mapping_validation.json 标注 static_schema_only 正常，它不是训练验证。
- report.md 和 report.json 成功生成，输出严格复核完成。
- 原始数据保持不变；失败演示仍保留。需要成功子集时应另做筛选实验。

## 清洗完成后接入 LingBot（可选后续）

复用 Linux LingBot 环境，按上游说明准备依赖；归一化使用当前项目的单 GPU 启动器：
```bash
cd "$PROJECT"
python -m scripts.run_lingbot_norm \
  --lingbot-root /code/lingbot-vla \
  --dataset "$OUTPUT" --cuda-devices 0 --dry-run
# 核对路径后，去掉 --dry-run 执行。
```

启动器会拒绝已有统计文件，并检查新统计的映射字段、形状、有限性和完整帧数。
输出文件系统需支持硬链接，以便原子发布已验证文件。
正式训练前以演示为单位设计训练/验证划分，仅用训练数据计算统计；本工具目前没有新增分割/筛选功能。
先验证一批数据的三个视角、7+1 个有效动作维度、填充 mask 和任务文本，再运行少量前向/反向步骤。

上游依据：
- https://github.com/Robbyant/lingbot-vla/blob/main/README.md
- https://github.com/Robbyant/lingbot-vla/blob/main/scripts/compute_norm.py
