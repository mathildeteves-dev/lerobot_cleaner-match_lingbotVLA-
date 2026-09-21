"""Typer CLI entry point for lerobot-cleaner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from lerobot_cleaner.v21.config import CleaningConfig
from lerobot_cleaner.v21.inspector import inspect_dataset
from lerobot_cleaner.v21.pipeline import Pipeline
from lerobot_cleaner.v21.reader import LeRobotDataset
from lerobot_cleaner.v21.validate import InputContractError, validate_dataset

app = typer.Typer(
    help="GR00T v2.1 cleaning and conservative LeRobot v3.0 cleaning.",
    add_completion=False,
)
console = Console()

DEFAULT_CONFIG_NAME = "cleaning_config.yaml"


@app.command()
def inspect(dataset: Path = typer.Argument(..., help="Path to a GR00T LeRobot dataset")):
    """Scan a dataset and print a summary."""
    ds = LeRobotDataset(dataset)
    summary = inspect_dataset(ds)
    console.print_json(json.dumps(summary.to_dict(), ensure_ascii=False))


@app.command("check")
def check(dataset: Path = typer.Argument(..., help="Path to a GR00T LeRobot dataset")):
    """Validate that a dataset meets the GR00T-format LeRobot v2.1 input contract."""
    try:
        warnings = validate_dataset(dataset)
    except InputContractError as e:
        console.print(f"[red]✗ Input contract not met[/red]\n{e}")
        raise typer.Exit(code=1)
    console.print(f"[green]✓ {dataset} meets the GR00T-format LeRobot v2.1 contract.[/green]")
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")


@app.command()
def run(
    dataset: Path = typer.Argument(..., help="Path to the input dataset"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to cleaning_config.yaml"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dataset path"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report only, no output"),
    num_workers: int = typer.Option(8, "--num-workers", "-j"),
    resume: bool = typer.Option(False, "--resume", help="Skip already-written episodes"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip wizard even if no config"),
):
    """Clean a dataset (Mode A: config-driven; Mode B: interactive wizard)."""
    info_path = dataset / "meta/info.json"
    if (
        info_path.is_file()
        and json.loads(info_path.read_text(encoding="utf-8")).get("codebase_version") == "v3.0"
    ):
        console.print(
            "This dataset is already v3.0. Use audit-v3 / clean-v3, not the GR00T v2.1 run command."
        )
        raise typer.Exit(code=1)
    cfg = _resolve_config(dataset, config, output, preset, dry_run, num_workers, resume, yes)
    if cfg is None:
        raise typer.Exit(code=1)

    console.print(f"[bold]Cleaning[/bold] {cfg.input} → {cfg.output} (dry_run={cfg.dry_run})")
    try:
        pipeline = Pipeline(cfg)
    except InputContractError as e:
        console.print(f"[red]✗ Input contract not met[/red]\n{e}")
        raise typer.Exit(code=1)
    for w in pipeline.input_warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    report = pipeline.run()
    report.print_summary()
    report_dir = Path(cfg.output) / "cleaning_report"
    console.print(f"[green]Report:[/green] {report_dir / 'report.md'}")


def _resolve_config(dataset, config, output, preset, dry_run, num_workers, resume, yes):
    dataset = Path(dataset)

    # Mode A: explicit config, or auto-discovered config in dataset dir.
    if config is None:
        candidate = dataset / DEFAULT_CONFIG_NAME
        if candidate.exists():
            config = candidate

    if config is not None:
        cfg = CleaningConfig.from_yaml(config)
        cfg.input = cfg.input or dataset
        if output:
            cfg.output = output
        if cfg.output is None:
            cfg.output = Path(str(dataset).rstrip("/") + "_clean")
        cfg.dry_run = dry_run or cfg.dry_run
        cfg.num_workers = num_workers
        cfg.resume = resume or cfg.resume
        return cfg

    # Mode B: wizard (unless --yes => defaults).
    out = output or Path(str(dataset).rstrip("/") + "_clean")
    if yes:
        cfg = CleaningConfig(input=dataset, output=out, dry_run=dry_run, num_workers=num_workers)
        if preset:
            cfg.preset = preset
            from lerobot_cleaner.v21.config import load_preset

            cfg = cfg.merge_preset(load_preset(preset))
            cfg.input, cfg.output = dataset, out
        return cfg

    from lerobot_cleaner.wizard import run_wizard

    cfg = run_wizard(dataset, out)
    cfg.dry_run = dry_run
    cfg.num_workers = num_workers
    return cfg


@app.command()
def validate(config: Path = typer.Argument(..., help="Path to a cleaning_config.yaml")):
    """Validate a config file against the schema."""
    cfg = CleaningConfig.from_yaml(config)
    console.print("[green]Config is valid.[/green]")
    console.print_json(cfg.model_dump_json())


@app.command("export-lingbot")
def export_lingbot(
    dataset: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Path to the cleaned LeRobot v2.1 dataset",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination path for the LeRobot v3.0 dataset",
    ),
    data_name: Optional[str] = typer.Option(
        None,
        "--data-name",
        help="LingBot data name; must match the robot-config filename",
    ),
    robot_config: Optional[Path] = typer.Option(
        None,
        "--robot-config",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="LingBot robot feature-mapping YAML",
    ),
):
    """Export a cleaned LeRobot v2.1 dataset for LingBot-VLA (LeRobot v3.0)."""
    # Import lazily so other CLI commands do not require the optional LeRobot package.
    from lerobot_cleaner.v30.lingbot.vla import export_lingbot_dataset

    console.print(f"[bold]Exporting for LingBot-VLA[/bold] {dataset} -> {output}")
    try:
        result = export_lingbot_dataset(
            dataset=dataset,
            output=output,
            data_name=data_name,
            robot_config=robot_config,
        )
    except (FileExistsError, ImportError, RuntimeError, ValueError) as e:
        console.print(f"[red]Export failed:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"[green]LingBot-VLA dataset:[/green] {result}")
    if robot_config is not None:
        console.print(f"[green]Robot config:[/green] {robot_config}")


@app.command("audit-v3")
def audit_v3_command(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    verify_videos: bool = typer.Option(False, "--verify-videos"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True),
):
    """Read-only v3.0 structure, numeric and optional video decode audit."""
    from lerobot_cleaner.v30.v3 import V3Config, audit_v3

    try:
        cfg = V3Config.from_yaml(config)
        cfg.verify_videos = cfg.verify_videos or verify_videos
        console.print_json(json.dumps(audit_v3(dataset, cfg)))
    except (OSError, ValueError, KeyError, ImportError) as e:
        console.print(f"Audit failed: {e}", markup=False)
        raise typer.Exit(code=1) from e


@app.command("calibrate-v3")
def calibrate_v3_command(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    quantile: float = typer.Option(0.995, "--quantile"),
    mad_k: float = typer.Option(8.0, "--mad-k"),
    min_samples: int = typer.Option(20, "--min-samples", min=2),
    clean_output: Optional[Path] = typer.Option(None, "--clean-output"),
):
    """Read-only group calibration; optionally clean using the generated thresholds."""
    from lerobot_cleaner.v30.calibration import CalibrationConfig, calibrate_v3
    from lerobot_cleaner.v30.v3 import V3Config

    try:
        settings = CalibrationConfig(quantile=quantile, mad_k=mad_k, min_samples=min_samples)
        report = calibrate_v3(dataset, output, V3Config.from_yaml(config), settings,
                              clean_output=clean_output)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        console.print(f"Calibration failed: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print(f"Calibration: {report['status']}; report: {output / 'calibration_report.json'}")
    if report["threshold_config"]:
        console.print(f"Threshold config: {report['threshold_config']}")
    if clean_output is not None:
        console.print(f"Cleaning: {report['cleaning']['status']}")
    if report["status"] != "ready":
        raise typer.Exit(code=2)


@app.command("clean-v3")
def clean_v3_command(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True),
    verify_videos: bool = typer.Option(False, "--verify-videos"),
    resume: bool = typer.Option(False, "--resume"),
):
    """Clean v3.0 numeric values without removing rows or changing videos."""
    from lerobot_cleaner.v30.v3 import V3Config, clean_v3

    try:
        cfg = V3Config.from_yaml(config)
        cfg.verify_videos = cfg.verify_videos or verify_videos
        report = clean_v3(dataset, output, cfg, resume=resume)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as e:
        console.print(f"Cleaning failed: {e}", markup=False)
        raise typer.Exit(code=1) from e
    console.print(f"Saved {report['frames']} frames / {report['episodes']} episodes to {output}")
    console.print(
        f"Changed values: {report['changed_values']}; videos: {report['video_verification']}"
    )
    console.print(f"Report: {output / 'cleaning_report/report.md'}")


@app.command("smoke-lingbot")
def smoke_lingbot_command(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    lingbot_root: Path = typer.Option(..., "--lingbot-root", exists=True),
    profile: Path = typer.Option(..., "--profile", exists=True),
    robot_config: Path = typer.Option(..., "--robot-config", exists=True),
    train_config: Path = typer.Option(..., "--train-config", exists=True),
    config: Optional[Path] = typer.Option(None, "--config", exists=True),
    cuda_device: str = typer.Option("0", "--cuda-device"),
    level: int = typer.Option(1, "--level", min=1, max=3, help="1: loader; 2: training batch; 3: one model forward"),
):
    """Real LingBot data-pipeline smoke; run from the repository in its Linux environment."""
    from scripts.smoke_lingbot import smoke
    try:
        result = smoke(dataset, output, lingbot_root, profile, robot_config, train_config, config, cuda_device, level=level)
    except (OSError, ValueError, ImportError) as exc:
        console.print(f"Smoke failed: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(result))
    if result["status"] != "passed":
        raise typer.Exit(code=2)


@app.command("validate-lingbot")
def validate_lingbot_command(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    robot_config: Path = typer.Option(..., "--robot-config", exists=True),
    train_config: Path = typer.Option(..., "--train-config", exists=True),
):
    """Validate mapping slices/cameras against dataset metadata (not a training test)."""
    from lerobot_cleaner.v30.lingbot.config import validate_mapping

    try:
        result = validate_mapping(dataset, robot_config, train_config)
    except (OSError, ValueError, KeyError, TypeError) as e:
        console.print(f"Mapping validation failed: {e}", markup=False)
        raise typer.Exit(code=1) from e
    console.print_json(json.dumps(result))


@app.command("generate-lingbot-config")
def generate_lingbot_config(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    spec: Path = typer.Option(..., "--spec", exists=True, help="Embodiment spec YAML"),
    robot_config: Optional[Path] = typer.Option(
        None,
        "--robot-config",
        help="Output robot mapping YAML (default configs/robot_configs/<name>.yaml under CWD)",
    ),
    train_config: Optional[Path] = typer.Option(
        None,
        "--train-config",
        help="Output train YAML (default configs/vla/<name>.yaml under CWD)",
    ),
):
    """Generate a LingBot robot config + train config pair from an embodiment spec."""
    from lerobot_cleaner.v30.lingbot.generate import generate_configs

    try:
        result = generate_configs(dataset, spec, robot_config, train_config)
    except (FileExistsError, OSError, ValueError, KeyError, TypeError) as e:
        console.print(f"Generation failed: {e}", markup=False)
        raise typer.Exit(code=1) from e
    console.print_json(json.dumps(result))


if __name__ == "__main__":
    app()
