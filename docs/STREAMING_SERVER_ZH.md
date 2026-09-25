# 完整 DROID 数据：96 GiB 容器上的分批清洗

> 当前统一入口已改为官方 LeRobotDataset → checks/plans → transforms → 官方 writer。
> 本文下方保留旧 native 引擎的服务器运行记录与操作背景，不能作为新版性能、续跑或测试结论。
> 新版完整契约以 [规则与变换](V3_RULES_AND_TRANSFORMS_ZH.md) 为准：
> - `memory` 和 `streaming` 都进入官方逐集 pipeline；官方 HF 初始化内存不受 batch_rows 限制。
> - 先完整生成报告和计划，再执行变换；报告和计划随集数/删帧数量增长。
> - 无修改时复制；有修改时官方重写视频、索引、metadata 和统计，可能改变分片与视频偏移。
> - 支持显式删帧、删集、静止裁剪、ROI、夹爪和数值变换；默认不启用这些结构修改。
> - --resume 校验来源和配置后重新执行，不复用旧数值/视频检查点或半写入 episode。
> - 空间预检为输入 meta/data/videos 总大小的三倍加保留空间，并非大小保证。
> - 本次只修改代码和文档，未运行程序、测试、编译或性能验证。

服务器排查补充：若原始目录混有未引用的重复索引分片，请使用
[重复分片处理步骤](STALE_SHARDS_ZH.md) 中的显式配置。不要直接修改 total_frames 或删除文件。

## 这次解决什么问题

服务器的数据为 95,658 集、27,630,375 帧；容器内存硬上限为 96 GiB。
宿主机 `free -h` 显示的可用内存不能当成容器额度，A800 的显存也不会被此清洗器使用。
旧的全量读取方式不能只改总帧数限制就认为安全。

本次增加 `engine: streaming`，通过 PyArrow 分批读入数据和逐集元数据，
不再把所有 parquet 转成 Pandas 后拼接。原 `memory` 引擎保留用于兼容和小样例对照。

已在本地真实 100 集样例上完成分批清洗、内容核对，以及超过 100 万帧的合成审计测试。
没有连接你的服务器运行完整数据，不能据此宣称全部 2,763 万帧已验证通过。

## 1. 先同步代码，不要只改 YAML

本次修改发生在本地项目，不会自动出现在 `/code` 下的服务器副本，也未自动推送 GitHub。
可将提供的更新 ZIP 上传到服务器，先备份/对比同名代码文件，然后在项目根目录解压合并。
更新包不包含数据、模型、虚拟环境或 Git 历史，不会覆盖这些目录。
服务器若有你另外修改过的同名文件，请先对比后合并，而不是直接覆盖。

主要代码文件：

- `lerobot_cleaner/v30/v3.py`
- `lerobot_cleaner/v30/v3_streaming.py`（新增）
- `lerobot_cleaner/v30/v3_stream_stats.py`（新增）
- `lerobot_cleaner/cli.py`
- `scripts/run_droid_clean.py`
- `configs/cleaning/droid_v3.yaml`
- `tests/test_v3_streaming.py`（新增，复用原 tests/test_v3.py 的合成样例）

也可通过自己的 Git 提交/推送流程同步；只有确实推送过这些修改，服务器 `git pull` 才能拿到它们。

激活原清洗环境并进入项目目录：

```bash
cd /code/lerobot_cleaner-match_lingbotVLA-
source .venv-cleaner/bin/activate
python -m pip install -e ".[dev,lingbot]"
python -m scripts.run_droid_clean --help
```

帮助信息应该出现 `--resume`、`--quiet`。核对实际配置：

```bash
python -c "from pathlib import Path; from lerobot_cleaner.v30.v3 import V3Config; c=V3Config.from_yaml(Path('configs/cleaning/droid_v3.yaml')); print(c.model_dump())"
```

必须看到 `engine: streaming`（Python 字典中显示 `'engine': 'streaming'`）和 `batch_rows: 8192`。
本次没有新增必装依赖，PyArrow/NumPy/Pandas 已在项目基础依赖中；不需要安装 GPU 清洗库。

## 2. 先运行完整只读审计

```bash
DATASET="/data/bai0111/droid_1_0_1_lerobotv3"
OUTPUT="/data/bai0111/droid_1_0_1_clean_v3"

python -m scripts.run_droid_clean --dataset "$DATASET" --audit-only
```

这是分批扫描全数据，不写清洗结果；会显示进度，最后打印规模、失败演示数、
非有限值计数和映射静态检查结果。只读审计遇到 NaN/Inf 会报告数量；
正式清洗在默认 `nonfinite: error` 下会停止，需要先决定修复策略。

如果遇到索引不一致、相机不存在、字段切片不匹配、视频缺失或超长演示，先停止排查。
不会因为本地样例通过，就自动忽略完整数据里的不同情况。

## 3. 审计通过后清洗

```bash
python -m scripts.run_droid_clean --dataset "$DATASET" --output "$OUTPUT"
```

程序依次显示数值清洗、视频复制/校验和输出复核进度。
正式清洗在处理数值时同步做输入检查，不会在内部先重复运行一遍全量输入审计。
最后的输出复核仍会分批读取完整输出，属于必要的校验阶段。

原始数据不变。所有行、失败标记、额外字段、数据分文件的行数边界、逐集视频偏移均保留。
一段演示横跨读取批次或多个数据文件时也会完整处理，不按批次边界截断或跨集插值。
本次不新增删帧、删集、裁剪、二值化、成功率筛选或视频重编码策略。

## 4. 内存、统计与保护参数

默认文件：`configs/cleaning/droid_v3.yaml`。它现在指定 `reader_backend: lerobot`，
在 LingBot 环境使用官方 LeRobot 0.4.2。官方 HF loader 初始化/缓存的内存不受 `batch_rows` 限制。
统一入口已移除 native 后端，不能通过修改配置切换回旧读取器。
下文历史 native 测量不适用于当前官方 loader。
官方读取异常不会自动切换后端。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| engine | streaming | 必须使用新分批引擎，不是仅放宽旧引擎限制 |
| max_frames | 30,000,000 | 数据集总规模的误操作保护，不是一次加载行数 |
| batch_rows | 8,192 | cleaner 写入缓冲大小，不限制官方 HF 初始化 |
| metadata_batch_rows | 64 | 逐集元数据的读取/写出批次大小 |
| max_episode_frames | 100,000 | 为跨批次插值/逐集统计保留的单集行数上限 |
| quantile_samples | 32,768 | 每个数值字段的固定容量全局分位数抽样 |
| disk_reserve_gb | 5 | 磁盘容量预检的额外保留空间，单位 GiB |

native 后端的清洗缓冲主要随批次、单集大小、数值维数、抽样容量变化，不随总帧数线性增长；此结论不能套用于官方 HF loader 初始化。
仍会保留紧凑的文件清单、视频区间清单；它们随文件数/集数增长，但不包含全部帧或全部逐集统计。
PyArrow 解压缓冲、字符串长度和 Python 分配器仍会影响真实峰值，因此不是硬性的内存额度保证。
异常巨大的单集会被拒绝，而不是无限累积内存。

可以另开一个终端观察容器内存：

```bash
watch -n 5 'cat /sys/fs/cgroup/memory/memory.usage_in_bytes'
```

这里输出字节数；上限是 `103079215104`。同容器的其他进程也会共享额度。
如出现异常增长，可正常中断，再检查原因或降低批次参数；不要只继续放大各项上限。

统计方法变化：

- 全局 mean/std/min/max/count 使用全部有效输出帧，采用稳定的分批均值/方差合并，不做抽样。
- 逐集分位数通过该集的完整数值计算，保持精确计算方式。
- 全局 q01/q10/q50/q90/q99 从均匀 reservoir 抽样估计，默认最多 32,768 行/字段、随机种子 0。
  只有全部帧能放进抽样容量时才与全量分位数等价；完整 DROID 的全局分位数是近似值。
- 抽样使用向量化 Algorithm R，不会对 2,763 万帧逐行运行 Python 抽样循环。
- 视频保持原样，沿用原图像统计；LingBot 归一化仍必须用 LingBot 自己的脚本计算。

报告包含 `global_quantiles`、`peak_input_batch_rows`、`peak_episode_rows`，
明确记录统计方法和实际处理块大小；后两项不是 RAM 测量值。

## 5. 磁盘与视频

仍然复制视频，不默认使用硬链接/软链接，避免之后修改源视频影响清洗副本。
复制后核对 SHA256。373 GB 视频的复制和哈希验证会产生较多磁盘读取，进度耗时取决于存储速度。
容量预检估算为“剩余视频 + 3 倍原数据/元数据文件大小 + 额外保留空间”，
用于尽早拒绝明显空间不足，但不是实际压缩后大小的保证，也无法发现所有平台配额。

默认 `metadata_only` 不等于像素逐帧解码通过。完整视频检查需额外安装 PyAV：

```bash
python -m pip install -e ".[v3-video]"
python -m scripts.run_droid_clean --dataset "$OUTPUT" --audit-only --verify-videos
```

新的解码检查逐帧计数，不保存整条视频的时间戳数组。但完整视频解码可能很慢，
本次只对它做了模拟解码器的逻辑测试，没有在此环境实际解码你的 AV1 视频。

## 6. 中断与续跑的准确边界

运行中的数据存放在 `$OUTPUT.partial/dataset`，通过全部检查后才移至 `$OUTPUT`。
正常完成后 `.partial` 中仍保留少量检查点用于溯源，不会再保留一份完整数据。
不要将 `.partial/dataset` 当成已完成的训练数据。

中断后，用相同输入、输出和配置执行：

```bash
python -m scripts.run_droid_clean --dataset "$DATASET" --output "$OUTPUT" --resume
```

- 数值阶段尚未完成：先删除本任务暂存的未完成 data/meta，再从头分批扫描这一阶段。
  这是阶段级恢复，**不是从上次的某一帧接着计算**；不会删除原数据。
- 数值阶段已完成：验证已写数据/元数据的哈希，通过后复用，不重新清洗。
- 已复制视频：再次核对副本哈希，通过则跳过复制；损坏副本会重新复制。
- 输入目录、路径/文件大小/修改时间指纹或配置改变：拒绝复用检查点，需使用新的输出目录。
  不要在运行中改动输入；指纹不是对全部输入内容的加密哈希证明。
- 已完成输出存在：始终拒绝覆盖，不能再用 `--resume` 覆盖它。

同一输出目录禁止并行运行。正常异常/中断会释放 `.partial/.lock`；
如果进程被 OOM 或 SIGKILL 强制杀死，锁可能残留。
只有确认旧进程已经停止后才能手动删除该锁，再执行 `--resume`；不要删除整个原始数据或随意清理检查点。
改变 `--quiet` / `--verify-videos` 等影响实际配置的选项也会改变本次任务身份，续跑请保持一致。

## 7. 查看结果与验证范围

```bash
cat "$OUTPUT/cleaning_report/report.md"
python -m pytest -q
python -m ruff check .
```

`report.json` 保存输入/输出数值摘要、处理块大小、统计方法、视频哈希与恢复策略。
本地 100 集样例通过全部数据列和非统计元数据逐项对照，保持数值不变。
百万帧合成测试验证了固定批次读取、跨文件演示及禁止全量 Pandas 读取/拼接。
中断恢复测试覆盖配置/源数据变化、损坏检查点和损坏视频副本。
这些验证不等于在你的完整服务器数据上已跑通，也不等于 LingBot 训练和真机控制已经验证。

本次最终回归：72 项通过、6 项因缺少 ffmpeg 跳过；Ruff 检查通过。
百万帧合成审计实际处理 1,001,472 帧，测试批次上限 2,048 行、单集 512 行。
此前真实 100 集样例使用 native 后端，最大输入批次 8,192 行、最长单集 1,627 行，改动数值为 0。
