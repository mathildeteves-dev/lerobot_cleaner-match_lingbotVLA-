"""Typer CLI entry point for lerobot-cleaner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from lerobot_cleaner.config import CleaningConfig
from lerobot_cleaner.dataset.inspector import inspect_dataset
from lerobot_cleaner.dataset.reader import LeRobotDataset
from lerobot_cleaner.dataset.validate import InputContractError, validate_dataset
from lerobot_cleaner.pipeline import Pipeline

app = typer.Typer(
    help="Configurable cleaning tool for GR00T-format LeRobot datasets.",
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
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to cleaning_config.yaml"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output dataset path"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report only, no output"),
    num_workers: int = typer.Option(8, "--num-workers", "-j"),
    resume: bool = typer.Option(False, "--resume", help="Skip already-written episodes"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip wizard even if no config"),
):
    """Clean a dataset (Mode A: config-driven; Mode B: interactive wizard)."""
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
            from lerobot_cleaner.config import load_preset

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


if __name__ == "__main__":
    app()
