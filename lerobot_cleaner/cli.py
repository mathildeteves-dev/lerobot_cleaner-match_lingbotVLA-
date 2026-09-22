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
    help="Official LeRobot loading, semantic adapters and quality evaluation; v2.1 converts to v3.0.",
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


@app.command("run-v21-legacy")
def run_legacy(
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


@app.command("run")
def run(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume: bool = typer.Option(False, "--resume"),
):
    """Unified official loader entry; v2.1 requires converted_root in config."""
    from lerobot_cleaner.v30.v3 import V3Config, audit_v3, clean_v3
    try:
        cfg = V3Config.from_yaml(config)
        if dry_run:
            if resume:
                raise ValueError("--resume is only supported for cleaning")
            report = audit_v3(dataset, cfg)
        else:
            if output is None:
                raise ValueError("Provide --output for cleaning, or --dry-run for quality evaluation")
            report = clean_v3(dataset, output, cfg, resume=resume)
        console.print_json(json.dumps(report))
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        console.print(f"Run failed: {exc}", markup=False)
        raise typer.Exit(code=1) from exc


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
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as e:
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
    """Execute v3 transform plans and rebuild a consistent official dataset."""
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




@app.command("check-training")
def check_training_command(
    dataset: Path = typer.Argument(..., exists=True, file_okay=False),
    target: str = typer.Option("lingbot", "--target"),
    robot_config: Path = typer.Option(..., "--robot-config", exists=True),
    train_config: Path = typer.Option(..., "--train-config", exists=True),
    config: Optional[Path] = typer.Option(None, "--config", exists=True, help="Official loader / v2.1 conversion config"),
    output: Optional[Path] = typer.Option(None, "--output", help="New report JSON outside the dataset"),
    entrypoint: str = typer.Option("official_train", "--entrypoint"),
    padding_warning_ratio: Optional[float] = typer.Option(None, "--padding-warning-ratio", min=0, max=1),
    decode_cameras: bool = typer.Option(False, "--decode-cameras"),
    camera_sample_stride: int = typer.Option(30, "--camera-sample-stride", min=1),
    padding_by_timestep: bool = typer.Option(False, "--padding-by-timestep"),
    tokenize: bool = typer.Option(False, "--tokenize", help="Check all distinct tasks with the real local tokenizer"),
    tokenizer_path: Optional[str] = typer.Option(None, "--tokenizer-path", help="Local processor directory or cached model ID"),
    smoke: bool = typer.Option(False, "--smoke", help="Real preprocessing/collator, no model weights"),
    lingbot_root: Optional[Path] = typer.Option(None, "--lingbot-root", exists=True),
    norm_stats: Optional[Path] = typer.Option(None, "--norm-stats", exists=True),
):
    """Diagnose model-specific readiness separately from dataset quality."""
    from lerobot_cleaner.training.config import TrainingCheckConfig
    from lerobot_cleaner.training.compatibility.lingbot import check_training
    from lerobot_cleaner.v30.v3 import V3Config
    try:
        if output and (output.exists() or output.resolve().is_relative_to(dataset.resolve())):
            raise ValueError("Report must be new and outside source dataset")
        if tokenizer_path and not tokenize:
            raise ValueError("--tokenizer-path requires --tokenize")
        if smoke and (lingbot_root is None or norm_stats is None):
            raise ValueError("--smoke requires --lingbot-root and --norm-stats")
        options = TrainingCheckConfig(target=target, robot_config=robot_config.resolve(), train_config=train_config.resolve(),
            entrypoint=entrypoint, padding_warning_ratio=padding_warning_ratio, decode_cameras=decode_cameras,
            camera_sample_stride=camera_sample_stride, include_padding_by_timestep=padding_by_timestep,
            tokenization={"enabled": tokenize, "tokenizer_path": tokenizer_path})
        loader_config = V3Config.from_yaml(config)
        if output and loader_config.converted_root and output.resolve().is_relative_to(loader_config.converted_root):
            raise ValueError("Report must be outside the converted source dataset")
        readiness = check_training(dataset, options, loader_config)
        if smoke:
            from lerobot_cleaner.training.smoke.lingbot import smoke_preprocessing
            readiness["smoke"] = smoke_preprocessing(Path(readiness.get("dataset", dataset)), options, readiness, lingbot_root, norm_stats)
            readiness["runtime_validated"] = readiness["smoke"]["status"] == "passed"
            readiness["training_ready"] = readiness["runtime_validated"]
        report = {"dataset_quality": {"status": "not_evaluated", "note": "Use audit-v3 for independent quality checks"},
                  "training_readiness": readiness}
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        console.print_json(json.dumps(report, ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
        console.print(f"Training compatibility check failed: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    if not readiness.get("compatible") or (smoke and not readiness["runtime_validated"]):
        raise typer.Exit(code=2)


@app.command("export-lingbot-bundle")
def export_lingbot_bundle(
    dataset: Path = typer.Option(..., "--dataset", exists=True, file_okay=False),
    output: Path = typer.Option(..., "--output"),
    lingbot_root: Path = typer.Option(..., "--lingbot-root", exists=True, file_okay=False),
    train_config: Path = typer.Option(..., "--train-config", exists=True, dir_okay=False),
    robot_config: Optional[Path] = typer.Option(None, "--robot-config", exists=True, dir_okay=False),
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True),
    skip_clean: bool = typer.Option(False, "--skip-clean"),
    skip_smoke: bool = typer.Option(False, "--skip-smoke", help="Export unverified artifacts; never marks training-ready"),
    cuda_device: str = typer.Option("0", "--cuda-device"),
):
    """Clean, validate, run official normalization and Level 2 smoke, then export a bundle."""
    from lerobot_cleaner.v30.lingbot.bundle import LingBotBundleExporter
    from lerobot_cleaner.v30.v3 import V3Config
    try:
        result = LingBotBundleExporter().export(dataset, output, lingbot_root=lingbot_root,
            train_config=train_config, robot_config=robot_config, config=V3Config.from_yaml(config),
            skip_clean=skip_clean, skip_smoke=skip_smoke, cuda_device=cuda_device)
        console.print_json(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        console.print(f"LingBot bundle export failed: {exc}", markup=False)
        raise typer.Exit(code=1) from exc
    if not result["training_ready"]:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
