# lerobot-cleaner

> A configurable, modular cleaning tool for **GR00T-format LeRobot datasets**.

Converting raw robot data to LeRobot format is only step one of VLA post-training
(e.g. NVIDIA Isaac GR00T). The tedious part is **cleaning** — and it differs per
embodiment and per task. `lerobot-cleaner` makes cleaning declarative: you pick
rules and parameters in a yaml file (or an interactive wizard), and the tool
produces a new clean dataset plus a cleaning report. It never mutates the input.

## Install

```bash
pip install -e ".[all]"      # includes matplotlib + opencv + dev
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
