# 重复索引分片：按逐集元数据选择数据文件

## 已确认的服务器现象

用户提供的只读检查结果显示：

- 汇总与逐集元数据均为 95,658 集、27,630,375 帧，区间连续、无重复演示编号。
- data 下 156 个文件总计 39,995,463 行，全部可读。
- 文件内部 index 连续，文件之间仅在 file-086 处回退一次。
- file-000～file-085 连续覆盖 0～27,630,374。
- file-086～file-155 额外覆盖 15,253,563～27,618,650，共 12,365,088 行。
- 逐集元数据只引用前 86 个文件，不引用后 70 个文件。

因此，后 70 个文件是当前逐集元数据未使用的重复索引分片。
这不证明它们的动作等所有内容与前面的分片相同，也不能确定它们最初为何出现。
不需要删除这些原始文件，也不能把 total_frames 改成 39,995,463。

## 本次新增的处理方式

原配置 `configs/cleaning/droid_v3.yaml` 仍使用默认 `data_file_policy: strict`，
所有匹配文件都计入，遇到不一致继续报错，不会默默跳过数据。

本次明确启用的配置是 `configs/cleaning/droid_v3_referenced.yaml`，其中：

```yaml
engine: streaming
data_file_policy: metadata_referenced
```

新策略从逐集元数据的 `data/chunk_index`、`data/file_index` 和 info 中的路径模板确定文件集合，
不是按固定文件编号截断，也不使用“达到总帧数就停止”的办法。
所选文件的总行数必须等于声明值；随后仍检查每一帧的 index、episode_index、frame_index、
时间戳、对应数据文件、特征与视频区间。不填造缺失行、不重写元数据引用、不自动去重合并数值。

如果所选文件缺失、存在内部重复/断裂，或其覆盖不完整，会失败并停止发布输出。
特别是某些数据集的一段演示跨进了“没有任何演示起点引用”的后续文件：
该策略会因总行数不足而拒绝，不会猜测应补入哪个分片。普通完整数据可继续用 strict。

## 更新服务器并运行

修改仍在本地，尚未推送 GitHub，也没有直接改服务器文件。
上传 `streaming-referenced-update.zip`，先备份/对比服务器上同名代码，再按目录结构合并到
`/code/lerobot_cleaner-match_lingbotVLA_2.0`。不能只上传 YAML，必须同步 Python 代码。
更新包是代码/配置/测试/文档，不包含数据或虚拟环境。

无需重建清洗环境，使用现有 `.venv-server` 即可：

```bash
cd /code/lerobot_cleaner-match_lingbotVLA_2.0
source .venv-server/bin/activate

DATASET="/data/bai0111/droid_1_0_1_lerobotv3"
OUTPUT="/data/bai0111/droid_1_0_1_clean_v3_referenced"
CONFIG="configs/cleaning/droid_v3_referenced.yaml"

# 先检查配置是否确实更新到这个 Python 环境
python -c "from pathlib import Path; from lerobot_cleaner.v30.v3 import V3Config; print(V3Config.from_yaml(Path('configs/cleaning/droid_v3_referenced.yaml')).data_file_policy)"

# 再进行完整只读分批审计
python -m scripts.run_droid_clean --dataset "$DATASET" --config "$CONFIG" --audit-only
```

配置检查应输出 `metadata_referenced`。根据此前文件头结果，审计应选择 86/156 个文件、
排除 70 个文件，其可读文件头行数合计为 12,365,088；处理的有效数据仍是 27,630,375 帧。
这些是预期，不代表本地已经读取过服务器所有有效数据。
如出现新的字段、索引、别名或视频错误，先停下来排查，不要跳过。

完整审计无异常后执行：

```bash
python -m scripts.run_droid_clean --dataset "$DATASET" --config "$CONFIG" --output "$OUTPUT"
cat "$OUTPUT/cleaning_report/report.md"
```

输出只包含选定的数据 parquet 和实际引用的视频；最终输出还会在 strict 策略下复核，
确保输出目录本身不依赖“忽略额外文件”才能被识别。
原始目录中的所有文件保持不变。数据集计数和视频偏移仍按原元数据保存。

输出报告 `cleaning_report/data_file_selection.json` 包含：

- 策略名称、发现的文件数、选中的文件路径。
- 每个排除文件的路径、原因、文件头行数；若排除文件损坏，则记录读取错误。
- 排除文件可读行数的合计。

`rows_removed: 0` 指选定数据内部没有删帧；不表示所有原始目录文件都复制到了输出。
排除分片与删帧分别记录。输入仍完整保留，可随时回头比对其他版本。

续跑需使用相同 CONFIG/路径并添加 `--resume`。不要尝试复用此前 strict 配置的检查点；
配置改变会被拒绝，应使用上面新的输出目录。其余内存、磁盘和恢复说明见
[分批清洗指南](STREAMING_SERVER_ZH.md)。

## 验证范围

新增测试覆盖：未引用文件保留原样、冲突动作内容不被混合、非固定编号选择、缺失引用文件、
所选数据索引仍被严格验证、不完整的跨文件覆盖被拒绝，以及排除清单记录。
本地测试不替代服务器完整数据验证；视频像素和 LingBot 训练仍须另行验证。
本次全套回归：79 项通过、6 项因缺少 ffmpeg 跳过；Ruff 检查通过。
本地 100 集真实样例也通过新配置的只读审计，没有额外分片时全部文件照常保留。
