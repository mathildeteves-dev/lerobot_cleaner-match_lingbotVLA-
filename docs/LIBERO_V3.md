# LIBERO-fastwam v3 清洗与质量复核

## 常用命令

在项目根目录运行：

```powershell
# 完整清洗：数值检查、视频全量解码、周期画面抽样和按任务预览。
python scripts/run_libero_clean.py --lingbot-root "D:/个人信息/实习/金鑫老师课题组/具身智能入门/lingbot-vla"
# 快速检查：不复制数据、不解码视频，明确标记视频验证不完整。
python scripts/run_libero_clean.py --audit-only --quick --output "../清洗结果/libero_audit_new"
# 完整只检查：包含视频解码/抽样；预览仅写报告目录，不写原始数据。
python scripts/run_libero_clean.py --audit-only --output "../清洗结果/libero_audit_full"
# 未完成的清洗任务续跑（输入、配置、profile、代码必须一致）。
python scripts/run_libero_clean.py --resume --lingbot-root "D:/个人信息/实习/金鑫老师课题组/具身智能入门/lingbot-vla"
```

默认输入：项目同级 `数据实例/lerobot_v30/libero_10_no_noops_lerobot`。
默认清洗输出：`清洗结果/libero_10_clean_v3_reviewed`。
默认只检查报告：`清洗结果/libero_10_audit`。
已有目录拒绝覆盖；复跑使用 `--output` 指定新目录。
旧 `libero_10_clean_v3` 和 DROID 的 `0916` 保持不变。

## 检查结论和退出状态

- `passed`：已执行的完整性检查通过，启发式质量检查未标记候选。
- `warning`：有待人工复核的轨迹/画面，或通过 `--quick` 跳过完整视频验证。退出码 0。
- `failed`：结构错误、任务文本缺失、非有限数值等。只检查模式同样明确打印异常字段和数量，退出码 2。
- `training_readiness=blocked`：清洗可以完成，但不能据此宣称可训练。单独列出控制语义、源码、运行依赖等缺口。

清洗时遇到结构或非有限数值等错误不会发布最终目录。异常摘要写到输出目录旁的 `.failure.json`（不会覆盖已有摘要）。
数值问题在只检查模式下可汇总到完整报告，`AUDIT_COMPLETE.json` 表示报告生成完成，其内部仍可能是 failed。

## Profile 与质量阈值

`--config` 控制通用清洗引擎，默认 `configs/cleaning/libero_v3.yaml`。
`--profile` 控制数据集字段和质量阈值，默认 `configs/profiles/libero_fastwam.yaml`。

Profile 配置字段维度、相机、状态/动作源、状态别名、浮点容差、语言关联和质量阈值。
可新增 profile 适配其他转换版本；不能仅因维度相同就假定动作控制语义相同。
默认别名容差为 atol=rtol=1e-6，允许微小浮点差异，保留原始值。

当前 LIBERO 数据结构：

- 388 条轨迹、104280 帧、20 FPS、10 条非空任务文本。
- 文本来自 `meta/tasks.parquet` 字符串索引（也支持显式 task 列），按 task_index 关联；不要求逐帧存在 language_instruction。
- observation.state 的前 6 维对应 ee_state，后 2 维对应 gripper_state。
- joint_state 为独立 7 维，不是 observation.state 的前 7 维。
- 原始 action 为 7 维，不做状态相减、单位转换、二值化或控制模式推断。
- 两路相机：observation.images.image、observation.images.wrist_image。
- 无成功标签，成功/失败数量未知。

逐轨迹指标：静止比例及最长静止时间、动作相邻帧变化、方向反复变化、轨迹时长、完全相同的状态/动作序列。
默认动作运动检查只使用前 6 维，排除最后一维正常的二值夹爪切换；此切片也可配置。
阈值使用原始数值单位，都是人工复核候选，**不自动删除轨迹或帧**。
完全相同的状态/动作序列不等于视频完全相同；当前不检测近似重复视频。

视频抽样在已有全量解码过程内执行，默认每秒检查一帧的小尺寸灰度图，记录黑屏、低细节、连续样本画面基本不变等候选。
低细节不等同于失焦，场景静止不等同于视频冻结；稀疏采样也不能排除采样间隙的问题。
每个任务优先选择一条轨迹保存中间帧，清洗时再加入数值候选轨迹，总数受 preview_limit 限制。
预览需要 Pillow，安装视频依赖可使用 `pip install -e ".[v3-video]"`。

## 自动报告与完成标记

清洗结果的 cleaning_report/ 自动包含：

- analysis_zh.md：中文摘要、任务分布、视频校验和训练缺口。
- review.html：可筛选候选轨迹的复核页面及抽查画面。
- episode_quality.json：逐轨迹指标、标记和异常片段帧号（最多列出前 20 个动作突跳位置，计数不截断）。
- video_quality.json、previews/：周期画面指标及预览。
- report.json：完整报告，含各视频实际解码帧数、区间帧数及是否复用检查点。
- profile.used.yaml、cleaning_config.used.yaml：实际生效配置。
- training_readiness.json：明确列出训练前尚未完成的验证。
- post_validation.json：元数据一致性、输出复扫等证据；数值摘要相同不冒充逐值相同。
- COMPLETE.json：全部必要复核及报告生成后，随整个输出目录一起发布。

报告包括 Python/依赖版本、Git 提交和当前 Python 源码哈希。只检查报告直接位于指定输出目录。
全局数值统计重算，当前 131072 行分位数容量覆盖 104280 帧，因此本数据集的分位数精确；更大数据集可能使用蓄水池抽样，报告注明。
图像统计沿用输入；输出画面不转码。

## 续跑与性能

数值质量检查合并到已有的按轨迹扫描，避免额外整表扫描。
单次内存中的数据帧仍受批大小/单集上限控制；逐轨迹摘要和去重索引的内存随轨迹数量增长。
数值阶段完成后可以复用；未完成的数值阶段仍从头开始。
视频复制完成后逐文件保存解码检查点，恢复时校验视频 SHA256、时间区间、检查参数和预览文件；一致时跳过已验证视频。
变更后的文件会重新解码。进度条显示文件及帧进度/预计剩余时间，`--quiet` 可关闭。
检查点在 `.partial` 同级任务目录，完成后保留以便追溯。
只检查模式暂不支持续跑；完整清洗模式支持。

## LingBot 实际加载检查

```powershell
python scripts/run_lingbot_probe.py --dataset "../清洗结果/libero_10_clean_v3_reviewed" --lingbot-root "D:/path/to/lingbot-vla" --output "../清洗结果/training_preflight_new.json"
# 完整环境中加上明确的映射和归一化文件：
python scripts/run_lingbot_probe.py --dataset "D:/path/to/cleaned" --lingbot-root "D:/path/to/lingbot-vla" --robot-config "D:/path/to/libero.yaml" --train-config "D:/path/to/train.yaml" --norm-stats "D:/path/to/norm.json" --profile "D:/path/to/verified_profile.yaml" --output "D:/path/to/probe.json"
```

检查命令调用实际 LingBot VLADataset，使用 getdata 避免 __getitem__ 失败后随机替换样本，读取首、中、末样本，验证任务文本、图像、动作序列、状态形状及有限值，并实际执行批次拼接。
提供归一化文件后会验证其结构及样本归一化。不会自动下载模型；缺少依赖/映射时输出阻塞原因，退出码 2。
数据加载通过时退出码 0，但报告仍不把数据加载等同于模型分词、视觉处理、训练前向或控制部署验证。

当前本机缺少 datasets、lerobot；本地 LingBot 评测使用轴角状态及 np.sign 夹爪处理，但下载数据末维为 0/1。
这些证据不足以确认发布者的转换约定，不会自动生成部署映射或把 profile.semantics.verified 改为 true。
该字段设为 true 时必须同时填写来源证据、动作模式、旋转表示、单位和夹爪约定。
