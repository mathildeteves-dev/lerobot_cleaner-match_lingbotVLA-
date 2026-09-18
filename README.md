# lerobot-cleaner

> GR00T v2.1 cleaning + conservative native LeRobot v3.0 cleaning with LingBot-VLA mapping support.

## LIBERO-fastwam v3

`libero_10_no_noops_lerobot` 请使用独立入口，不能沿用 DROID 的默认输入或关节切片。
在项目根目录执行 `python scripts/run_libero_clean.py`，默认输出到项目同级的
`清洗结果/libero_10_clean_v3_reviewed`。该入口按 profile 检查任务文本关联、状态结构、数值和视频，
自动生成中文报告、逐轨迹质量标记、画面预览及训练就绪检查；全量验证和报告生成后才发布结果。
保留全部帧并重算统计，支持视频解码续跑；不会生成未经验证的 LingBot 控制映射。
完整说明见 [LIBERO v3 清洗](docs/LIBERO_V3.md)。

## 本分支新增：DROID / LingBot-VLA

提供的 `droid_100_lerobotv3` 数据已经是 **LeRobot v3.0**，机器人是 **Franka 单臂**，
不是 R1Pro。请使用新增的 `clean-v3` 流程，不要对它执行旧版 `run` 或 `export-lingbot`。

- 新增 v3 元数据、数值和共享视频时间区间检查；可选 PyAV 完整解码检查。
- 新增保留所有帧的保守清洗：默认遇到非有限值报错，不删帧、不裁剪、不二值化、不转码。
- 新增 DROID 的 7 关节 + 1 夹爪、3 相机映射、训练起始配置及归一化启动脚本。
- 输出仍是 v3.0；重算数值统计，保持逐集索引和视频偏移，复制视频并校验 SHA256。
- `validate-lingbot` 是数据字段/配置的静态校验，不等于已经跑通 LingBot 训练。
- DROID 默认改用分批引擎：每批 8,192 行，单集上限 100,000 行，支持数值阶段/视频复制阶段续跑。
- 全局均值/方差等累计全部帧；全局分位数采用固定容量均匀抽样并在报告中注明，逐集分位数精确计算。

**完整数据（2,763 万帧 / 96 GiB 容器）请看 [服务器分批清洗步骤](docs/STREAMING_SERVER_ZH.md)。**
若遇到“156 个数据文件，但元数据只引用 86 个”的重复索引分片问题，见
[按元数据选择文件](docs/STALE_SHARDS_ZH.md)。新增配置必须显式指定；默认仍保持严格检查，不删除原始文件。
仅增大旧引擎 `max_frames` 不会降低内存占用；必须同步新的 Python 文件和 `engine: streaming` 配置。

详细说明、修改缘由与 Windows / Linux 操作见 [DROID 使用说明](docs/DROID_GUIDE_ZH.md)。

在本项目根目录执行（默认数据位于本项目同级的 `数据实例` 文件夹）：

```powershell
python -m pip install -e ".[dev]"
python -m scripts.run_droid_clean --audit-only
python -m scripts.run_droid_clean
```

默认输出到项目同级的 `清洗结果/droid_100_clean_v3`。已存在则拒绝覆盖；
再次运行时可用 `--output` 指定新目录。原始数据始终保留。

普通 v3 数据也可使用通用入口（在 Python 环境中安装本项目后）：

不传配置时保留旧的百万帧内存引擎；大数据必须传入分批配置。

```bash
lerobot-cleaner audit-v3 /absolute/path/to/data
lerobot-cleaner clean-v3 /absolute/path/to/data --output /absolute/path/to/new_output
# DROID streaming (run from this project root):
lerobot-cleaner audit-v3 /absolute/path/to/data --config configs/cleaning/droid_v3.yaml
lerobot-cleaner clean-v3 /absolute/path/to/data --config configs/cleaning/droid_v3.yaml --output /absolute/path/to/new_output
```

v3 路径暂不支持删集、删帧、重采样或旧版 R1–R7 规则，并非任意机器人数据的自动适配器。
以下原有文档描述 **GR00T v2.1 流程**，其能力与限制不要套用到新增 v3 路径。

## Original GR00T v2.1 workflow

Converting raw robot data to LeRobot format is only step one of VLA post-training
(e.g. NVIDIA Isaac GR00T). The tedious part is **cleaning** — and it differs per
embodiment and per task. `lerobot-cleaner` makes cleaning declarative: you pick
rules and parameters in a yaml file (or an interactive wizard), and the tool
produces a new clean dataset plus a cleaning report. It never mutates the input.

## Install

```bash
pip install -e ".[all]"     # matplotlib + opencv + dev
# Optional v2.1 -> v3.0 converter only (not needed for existing v3 data):
# pip install -e ".[all,lingbot]"
# Optional v3 video decode validation:
# pip install -e ".[v3-video]"
# requires ffmpeg/ffprobe in PATH for video rules
```

## Input requirements

lerobot-cleaner cleans datasets that are **already in GR00T-format LeRobot v2.1**.
It does **not** convert raw robot data, and it is not a general cleaner for any
LeRobot dataset — it relies on the GR00T-specific `meta/modality.json` to resolve
state/action keys. The tool is **dimension-agnostic**: state/action can be any
width, any number of arms, grippers, or camera views — all dims are read from
`modality.json`, nothing is hardcoded.

Your input directory must look like:

```
my_dataset/
├── meta/
│   ├── info.json          # v2.1: requires data_path, fps, chunks_size, features
│   ├── modality.json      # GR00T-specific: state/action key → {start, end}; video keys
│   ├── episodes.jsonl     # {episode_index, tasks, length} per line
│   ├── tasks.jsonl
│   └── stats.json         # optional (recomputed on output)
├── data/chunk-*/episode_*.parquet
└── videos/chunk-*/<video_key>/episode_*.mp4   # if video keys are declared
```

Each parquet must contain the columns: `observation.state`, `action`,
`timestamp`, `frame_index`, `episode_index`, `index` (state column **must** be
named `observation.state`, action **must** be `action`). The vector width of
`observation.state` / `action` must match the max `end` declared in
`modality.json`.

**Check your dataset before cleaning** — the preflight reports every problem at
once with an actionable message, and runs automatically at the start of `run`:

```bash
lerobot-cleaner check ./my_dataset
```

## Usage

### Mode A — config-driven

```bash
# Auto-discovers ./my_dataset/cleaning_config.yaml if present
lerobot-cleaner run ./my_dataset

# Or pass it explicitly
lerobot-cleaner run ./my_dataset --config ./my_config.yaml --output ./my_dataset_clean
```

### Mode B — interactive wizard

If no config is found, the wizard scans the dataset, asks per-rule questions, and
saves a `cleaning_config.yaml` you can re-edit later:

```bash
lerobot-cleaner run ./my_dataset
```

### Check / inspect / validate / dry-run

```bash
lerobot-cleaner check ./my_dataset             # preflight the input contract
lerobot-cleaner inspect ./my_dataset           # print a dataset summary
lerobot-cleaner validate ./my_config.yaml      # validate a config file
lerobot-cleaner run ./my_dataset --dry-run     # report only, no output
```

### Python API

```python
from lerobot_cleaner import clean
report = clean("./my_dataset", "./my_dataset_clean", config="my_config.yaml")
report.print_summary()
```

## Cleaning rules

| ID | Rule | What it does |
|----|------|--------------|
| R1 | `timestamp_alignment` | Monotonic timestamps, dt ≈ 1/fps, **uniform-dt check**, video↔row count |
| R2 | `static_frame_trim` | `trim_edges` (safe) or `drop_static_frames` (opt-in) |
| R3 | `gripper_binarize` | Threshold gripper dims; idempotent (skips already-binary) |
| R4 | `video_roi_crop` | Per-view ratio-based crop + resize (ffmpeg) |
| R5 | `episode_length_filter` | Drop too-short / too-long episodes |
| R6 | `numeric_sanity` | NaN/Inf, joint limits, **percentile outlier clipping** |
| R7 | `video_integrity` | Decodable + consistent frame counts across views |
| R8 | `reindex_and_restats` | Always last: reindex, rebuild meta, recompute stats |

Rule modules live under `lerobot_cleaner/v21/rules/`:

- `checks/integrity/`: `timestamp.py` (R1), `episode_length.py` (R5), `numeric.py` (legacy R6 facade)
- `checks/trajectory/`: `finite.py`, `joint_limits.py`, `velocity.py`, `acceleration.py`, `jerk.py`, `outlier.py`
- `transforms/numeric/`: `repair.py` (legacy repair policies), `percentile_clip.py`
- `checks/vision/video_integrity.py` (R7)
- `transforms/motion/static_trim.py` (R2)
- `transforms/embodiment/gripper.py` (R3)
- `transforms/vision/roi_crop.py` (R4)
- `finalizers/`: `reindex.py`, `lerobot_metadata.py`, `stats.py`, `lingbot_norm_stats.py` (R8; legacy `reindex_and_restats.py` facade retained)

Episode rules share `BaseRule.run(episode, context=None)`. `CheckRule.run()`
dispatches to `check()`; `TransformRule.run()` dispatches to `transform()`.
Existing class names and `apply(work)` calls remain supported. The stage runner
uses `run()` and forwards optional context to both stages. Checks may annotate or
reject, but must not mutate values/rows; this is an interface contract, not a runtime
copy/freeze mechanism. Dataset finalizers retain their separate dataset-level API.

Checks return `CheckResult(passed, rule, severity, metrics, message)` via `check()`,
`run()`, and the compatibility `apply()` entry point. `passed=False` records a detected
problem; `warning` allows processing to continue while configured rejection policies
set the episode's dropped state (`error`). Passing checks use `info`. Derivative checks
with insufficient samples return a warning with `evaluated: false`, not a pass.

Normal runs and dry-runs write `cleaning_report/episode_quality.jsonl`. Each record uses
the **source** `episode_index`, contains `dropped`, `drop_reason`, and `checks` keyed by
stable rule name. Each check preserves all five CheckResult fields (measurements stay
inside `metrics`). Results describe input data before transforms. Only executed checks
are present; disabled checks and checks skipped after rejection are absent. Derivative
metrics include per-target maxima, thresholds, and exceedance counts, plus top-level
`max_jerk`/`max_velocity`/`max_acceleration` and `threshold` for a single target. Missing
or non-finite numeric metrics serialize as JSON `null`. Records are sorted by source
index and include rejected episodes processed in this run.

Transforms return `TransformResult(episode, changed, metrics)` through `transform()`,
`run()`, and `apply()`. Each call reports its own changes, including no-ops; cumulative
rule counters remain available separately. `run()` records a lightweight summary on
`EpisodeWork.transform_results`. A replacement episode is adopted into the existing
work object so subsequent rules and legacy callers see its data, while preserving
prior check/transform history. No episode/DataFrame is serialized into these records.

R2 reports `frames_before`, `frames_after`, `trimmed_start`, `trimmed_end`, and
`trimmed_interior`, counted relative to that transform's input. Gripper transforms
report changed values per target; numeric transforms report repair/clip counts and
frame counts. ROI records describe deferred crop/resize instructions (encoding occurs
later in the worker). Both normal runs and dry-runs export these summaries under
`transforms` in `episode_quality.jsonl` and `episode_modifications` in `report.json`.
The Markdown report automatically summarizes episodes run/changed per transform.

The pipeline explicitly builds three stages:

- Checks: timestamp alignment → input episode length → video integrity → finite/joint limits/optional derivatives/outliers.
- Transforms (accepted episodes only): numeric repair → percentile clip → static trim → gripper binarization → ROI crop.
- Finalizers (once per dataset): `ReindexFinalizer` → `LeRobotMetadataFinalizer` → `StatsFinalizer` → optional `LingBotNormStatsFinalizer`.

Finalizers implement the separate `DatasetFinalizer.finalize(results, writer, report)`
contract, not `CheckRule` or `TransformRule`. Reindex publishes surviving episodes;
metadata writes episodes/tasks/info/modality and verifies video alignment; stats rereads
final parquet data and writes global/per-episode/relative statistics. The extra read
ensures statistics describe the final dataset. `DatasetWriter.finalize_meta()` remains
available for older callers that accumulate statistics during writes.

LingBot normalization is an optional integration hook, disabled by default. Python
callers may pass `lingbot_normalizer` to `build_finalizers`; the adapter receives the
final dataset path and must compute and validate LingBot statistics using the intended
robot mapping and training configuration. The existing `scripts/run_lingbot_norm.py`
is the v3 workflow; connecting it to v2.1 requires a compatible adapter. The hook refuses
empty datasets or known alignment errors; it never substitutes LeRobot statistics.

Checks short-circuit on rejection. Length limits now apply before transforms;
there is no second length filter after trimming. Checks only annotate or reject; they
never edit values or remove frames. Dry-run uses the same checks/transforms but skips
finalizers.

The existing `rules.numeric_sanity` configuration expands into separate checks and
transforms. `NumericSanityRule.apply(work)` remains a compatibility facade that runs
both stages. NaN/Inf interpolation, frame removal, and joint clipping live in
`NumericRepairRule`; quantile clipping lives in `PercentileClipRule`. Detection and
repair counters are reported under their respective rule names. All checks inspect
source values, so an episode configured for rejection can be rejected before repairs.
Repairs run before motion/gripper transforms and recompute masks after row removal;
an episode with no remaining rows is rejected. Percentile estimation excludes NaN/Inf;
dimensions with no finite samples have unconstrained percentile bounds.

Velocity, acceleration, and jerk are disabled by default. Enable them within
`rules.numeric_sanity` using `velocity`, `acceleration`, or `jerk` blocks containing
`enabled: true`, `limits: {state.arm: <positive threshold>}`, and optional
`on_violation: warn_only` (default) or `strict_drop`. The numeric_sanity master switch
must also be enabled. Targets use modality keys; thresholds are absolute component
magnitudes in target units per second, second squared, or second cubed. These checks
use repeated divided differences at interval midpoints, with source timestamps (or
source frame indices / fps when timestamps are absent). They do not unwrap angles or
convert units. Invalid timestamps/non-finite target values are reported or rejected;
episodes with too few samples are counted as insufficient, without estimating a derivative.


## GR00T correctness guarantees

This tool is built around what GR00T's data loader actually does
(`Isaac-GR00T/gr00t/data/dataset/lerobot_episode_loader.py`):

- **Video is indexed by integer frame number, positionally.** Frame `i` must equal
  parquet row `i`. R8 enforces `video_frame_count == row_count == episodes.jsonl
  length` per episode and **verifies it by re-opening every output video**.
- **Default normalization is min/max** (`use_mean_std=False`). A single outlier
  stretches the whole dataset's normalized range, so R6 offers percentile clipping.
- **Action chunks are built from consecutive rows assuming dt = 1/fps.**
  `drop_static_frames` breaks uniform dt, so it is opt-in and loudly reported;
  `trim_edges` (the default) preserves interior spacing. The writer always rebuilds
  a uniform timestamp column after cleaning.
- **Stats are recomputed over the global concatenation** via streaming accumulators
  (exact mean/std/min/max, reservoir-sampled q01/q99) — bounded memory at any scale.

## Dual-format output: GR00T **and** pi05/openpi, no extra processing

The cleaned dataset always trains on **both** frameworks out of the box, because
the two read normalization stats from different files (verified against each
loader's source):

- **GR00T** reads `meta/stats.json` (mean/std/min/max/q01/q99).
- **openpi / pi05** uses the upstream HF `lerobot` loader, which on v2.1 reads
  `meta/episodes_stats.jsonl` (per-episode min/max/mean/std/**count**).

We emit **both** files on every run. The per-episode stats are computed from the
already-loaded parquet rows — **no extra video decoding**, no measurable cost.
Image/video stats are intentionally omitted (GR00T's own `stats.json` omits them,
openpi normalizes images internally, and upstream `aggregate_stats` takes the key
union — so the output passes upstream `_assert_type_and_shape` + `aggregate_stats`
unchanged).

## Output layout

```
output_dataset/
├── meta/                       # standard GR00T LeRobot v2.1 meta (rewritten)
│   ├── info.json
│   ├── modality.json           # GR00T
│   ├── stats.json              # GR00T normalization stats
│   ├── episodes_stats.jsonl    # pi05/openpi normalization stats (+count)
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/chunk-*/               # reindexed parquet
├── videos/chunk-*/             # re-encoded, frame-aligned videos
└── cleaning_report/
    ├── report.md
    ├── report.json
    ├── cleaning_config.used.yaml
    └── figures/
```

## License

Apache-2.0


### v2.1 operational limits

Use a new, empty output directory for each normal run. Resume is currently rejected:
after dropping/reindexing episodes, output indices cannot reliably identify source
episodes. Dry-run writes reports only. Final statistics reject missing numeric columns,
dimension mismatches, and non-finite values; warning-only numeric policies do not make
invalid values suitable for training statistics. Video metadata dimensions are refreshed
from the first finalized video in each view after crop/resize. Video workflows require
both ffmpeg and ffprobe on PATH. LingBot normalization remains a configured adapter hook,
not a default operation of this v2.1 pipeline.
