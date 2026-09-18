"""Helpers for exporting cleaned datasets to the LingBot-VLA data format."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import yaml


def export_lingbot_dataset(
    dataset: Path,
    output: Path,
    data_name: str | None = None,
    robot_config: Path | None = None,
) -> Path:
    """Copy a local LeRobot v2.1 dataset and convert the copy to v3.0.

    LingBot-VLA keeps its robot feature mapping outside the dataset.  The
    Optional ``data_name`` / ``robot_config`` are retained for compatibility;
    format conversion itself does not need a robot mapping.
    """
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    robot_config = Path(robot_config).resolve() if robot_config is not None else None
    data_name = data_name.strip() if data_name is not None else None

    _validate_inputs(dataset, output, data_name, robot_config)

    try:
        from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
    except ModuleNotFoundError as e:
        raise ImportError(
            "LeRobot v0.4.2 is required. Install it with "
            "`pip install -e '.[lingbot]'` from the project directory."
        ) from e

    output.parent.mkdir(parents=True, exist_ok=True)

    # LeRobot's converter works in place and leaves a `<name>_old` sibling.
    # Running it in a temporary directory keeps both the input and destination
    # clean if conversion fails.
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.lingbot-", dir=output.parent
    ) as temp_dir:
        staged_dataset = Path(temp_dir) / output.name
        shutil.copytree(dataset, staged_dataset)

        convert_dataset(
            repo_id=staged_dataset.name,
            root=staged_dataset.parent,
            push_to_hub=False,
        )
        info = json.loads((staged_dataset / "meta/info.json").read_text(encoding="utf-8"))
        if info.get("codebase_version") != "v3.0":
            raise RuntimeError("Converter did not produce LeRobot v3.0")
        for required in ["meta/tasks.parquet", "meta/episodes", "data"]:
            if not (staged_dataset / required).exists():
                raise RuntimeError(f"Converter output is missing {required}")
        # The official converter replaces the staged folder. Restore auxiliary
        # reports from the untouched original, not from temporary `_old` state.
        if (dataset / "cleaning_report").is_dir():
            shutil.copytree(
                dataset / "cleaning_report", staged_dataset / "cleaning_report", dirs_exist_ok=True
            )
        if (dataset / "meta/modality.json").is_file():
            archive = staged_dataset / "cleaning_report"
            archive.mkdir(exist_ok=True)
            shutil.copy2(dataset / "meta/modality.json", archive / "source_modality.v21.json")
        shutil.move(str(staged_dataset), str(output))

    return output


def _validate_inputs(
    dataset: Path, output: Path, data_name: str | None = None, robot_config: Path | None = None
) -> None:
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    if not (dataset / "meta" / "info.json").is_file():
        raise ValueError(f"Not a LeRobot dataset (missing meta/info.json): {dataset}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output == dataset or dataset in output.parents:
        raise ValueError("Output must not be the input dataset or a directory inside it")
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            "export-lingbot requires v2.1. For existing v3.0 data use clean-v3; no conversion is needed."
        )
    for name in ["episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]:
        if not (dataset / "meta" / name).is_file():
            raise ValueError(f"Missing v2.1 metadata: meta/{name}")
    if data_name is None and robot_config is None:
        return
    if data_name is None or robot_config is None:
        raise ValueError("Provide both data-name and robot-config, or omit both")
    if not data_name:
        raise ValueError("data-name must not be empty")
    if not robot_config.is_file():
        raise ValueError(f"Robot config does not exist: {robot_config}")
    if robot_config.stem != data_name:
        raise ValueError(
            f"data-name '{data_name}' must match robot-config filename '{robot_config.name}'"
        )

    try:
        config = yaml.safe_load(robot_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as e:
        raise ValueError(f"Cannot read robot config {robot_config}: {e}") from e

    if not isinstance(config, dict):
        raise ValueError(f"Robot config must contain a YAML mapping: {robot_config}")
    missing_sections = {"states", "actions", "images"} - config.keys()
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"Robot config is missing section(s): {missing}")
