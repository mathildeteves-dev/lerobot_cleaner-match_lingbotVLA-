"""Interactive wizard: scan a dataset, ask per-rule questions, save a yaml config.

Used when no cleaning_config.yaml is found. Produces a CleaningConfig and writes
it to disk so the user can re-edit and re-run without the wizard next time.
"""

from __future__ import annotations

from pathlib import Path

import questionary
from rich.console import Console
from rich.table import Table

from lerobot_cleaner.v21.config import (
    CleaningConfig,
    GripperBinarizeConfig,
    GripperMode,
    NumericSanityConfig,
    OutlierMode,
    StaticFrameTrimConfig,
    StaticSource,
    StaticTrimMode,
)
from lerobot_cleaner.v21.inspector import DatasetSummary, inspect_dataset
from lerobot_cleaner.v21.reader import LeRobotDataset

console = Console()


def _print_summary(s: DatasetSummary) -> None:
    t = Table(title="Dataset Summary")
    t.add_column("field")
    t.add_column("value")
    t.add_row("robot_type", s.robot_type)
    t.add_row("codebase_version", s.codebase_version)
    t.add_row("fps", str(s.fps))
    t.add_row("episodes", str(s.total_episodes))
    t.add_row("frames", str(s.total_frames))
    t.add_row("state_dim", str(s.state_dim))
    t.add_row("action_dim", str(s.action_dim))
    t.add_row("state_keys", ", ".join(s.state_keys.keys()))
    t.add_row("video_keys", ", ".join(s.video_keys))
    t.add_row("length min/mean/max", f"{s.episode_length_min}/{s.episode_length_mean:.0f}/{s.episode_length_max}")
    console.print(t)


def run_wizard(input_path: Path, output_path: Path) -> CleaningConfig:
    ds = LeRobotDataset(input_path)
    summary = inspect_dataset(ds)
    _print_summary(summary)

    cfg = CleaningConfig(input=input_path, output=output_path)

    # R7 video integrity
    cfg.rules.video_integrity.enabled = questionary.confirm(
        "Enable video integrity checks (decodable + consistent frame counts)?", default=True
    ).ask()

    # R6 numeric sanity + outliers
    if questionary.confirm("Enable numeric sanity (NaN/Inf + outlier handling)?", default=True).ask():
        ns = NumericSanityConfig(enabled=True)
        mode = questionary.select(
            "Outlier handling mode (GR00T uses min/max norm — clipping is recommended):",
            choices=["warn", "clip_quantile", "drop_frame", "off"],
            default="warn",
        ).ask()
        ns.outlier_mode = OutlierMode(mode)
        cfg.rules.numeric_sanity = ns

    # R3 gripper binarize
    state_keys = list(summary.state_keys.keys())
    gripper_candidates = [k for k in state_keys if "grip" in k.lower()]
    if gripper_candidates and questionary.confirm(
        f"Binarize gripper dims {gripper_candidates}?", default=False
    ).ask():
        targets = []
        for k in gripper_candidates:
            targets.append(f"state.{k}")
            if k in summary.action_keys:
                targets.append(f"action.{k}")
        threshold = float(questionary.text("Threshold:", default="0.5").ask())
        mode = questionary.select("Mode:", choices=["binary_01", "binary_neg1_pos1"], default="binary_01").ask()
        cfg.rules.gripper_binarize = GripperBinarizeConfig(
            enabled=True, targets=targets, threshold=threshold, mode=GripperMode(mode)
        )

    # R2 static trim
    if questionary.confirm("Trim static frames?", default=True).ask():
        mode = questionary.select(
            "Mode (trim_edges is safe; drop_static_frames distorts dt):",
            choices=["trim_edges", "drop_static_frames"],
            default="trim_edges",
        ).ask()
        source = questionary.select("Judge motion from:", choices=["action", "state"], default="action").ask()
        pos_thr = float(questionary.text("Position threshold:", default="0.002").ask())
        cfg.rules.static_frame_trim = StaticFrameTrimConfig(
            enabled=True,
            mode=StaticTrimMode(mode),
            source=StaticSource(source),
            pos_threshold=pos_thr,
        )

    # R5 length filter
    if questionary.confirm("Filter episodes by length?", default=True).ask():
        cfg.rules.episode_length_filter.enabled = True
        cfg.rules.episode_length_filter.min_frames = int(
            questionary.text("Minimum frames:", default="30").ask()
        )

    # R1 timestamp alignment
    cfg.rules.timestamp_alignment.enabled = questionary.confirm(
        "Enable timestamp/alignment checks?", default=True
    ).ask()

    # R4 ROI crop is config-heavy; default off in wizard.
    console.print("[dim]Video ROI crop (R4) left disabled — edit yaml to enable.[/dim]")

    save_path = output_path.parent / "cleaning_config.yaml" if output_path else Path("cleaning_config.yaml")
    save_path = Path(questionary.text("Save config to:", default=str(save_path)).ask())
    cfg.to_yaml(save_path)
    console.print(f"[green]Saved config to {save_path}[/green]")
    return cfg
