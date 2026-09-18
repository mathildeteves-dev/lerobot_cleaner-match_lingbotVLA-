import json

import pytest
import yaml

from lerobot_cleaner.v30.lingbot.vla import _validate_inputs


def make_valid_inputs(tmp_path):
    """创建最小的测试数据集和机器人配置。"""
    dataset = tmp_path / "clean_v21"
    meta = dataset / "meta"
    meta.mkdir(parents=True)

    info = {
        "codebase_version": "v2.1",
        "fps": 15,
    }
    (meta / "info.json").write_text(
        json.dumps(info),
        encoding="utf-8",
    )
    for name in ["episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]:
        (meta / name).write_text("", encoding="utf-8")

    robot_config = tmp_path / "r1pro.yaml"
    robot_config.write_text(
        yaml.safe_dump(
            {
                "states": [],
                "actions": [],
                "images": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "clean_v30"
    return dataset, output, robot_config


def test_valid_inputs_are_accepted(tmp_path):
    dataset, output, robot_config = make_valid_inputs(tmp_path)

    _validate_inputs(
        dataset=dataset,
        output=output,
        data_name="r1pro",
        robot_config=robot_config,
    )


def test_data_name_must_match_robot_config_filename(tmp_path):
    dataset, output, robot_config = make_valid_inputs(tmp_path)

    with pytest.raises(ValueError, match="must match"):
        _validate_inputs(
            dataset=dataset,
            output=output,
            data_name="another_robot",
            robot_config=robot_config,
        )


def test_existing_output_is_rejected(tmp_path):
    dataset, output, robot_config = make_valid_inputs(tmp_path)
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        _validate_inputs(
            dataset=dataset,
            output=output,
            data_name="r1pro",
            robot_config=robot_config,
        )


def test_missing_robot_config_section_is_rejected(tmp_path):
    dataset, output, robot_config = make_valid_inputs(tmp_path)

    robot_config.write_text(
        yaml.safe_dump(
            {
                "states": [],
                "actions": [],
                # 故意缺少 images
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing section"):
        _validate_inputs(
            dataset=dataset,
            output=output,
            data_name="r1pro",
            robot_config=robot_config,
        )


def test_missing_dataset_info_is_rejected(tmp_path):
    dataset = tmp_path / "not_a_dataset"
    dataset.mkdir()

    robot_config = tmp_path / "r1pro.yaml"
    robot_config.write_text(
        yaml.safe_dump(
            {
                "states": [],
                "actions": [],
                "images": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing meta/info.json"):
        _validate_inputs(
            dataset=dataset,
            output=tmp_path / "output",
            data_name="r1pro",
            robot_config=robot_config,
        )


def test_conversion_does_not_require_mapping(tmp_path):
    dataset, output, _ = make_valid_inputs(tmp_path)
    _validate_inputs(dataset, output)


def test_already_v3_rejected(tmp_path):
    dataset, output, _ = make_valid_inputs(tmp_path)
    (dataset / "meta/info.json").write_text('{"codebase_version": "v3.0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="no conversion is needed"):
        _validate_inputs(dataset, output)


def test_export_preserves_report_and_original(tmp_path, monkeypatch):
    import sys
    import types
    from pathlib import Path

    from lerobot_cleaner.v30.lingbot.vla import export_lingbot_dataset

    dataset, output, _ = make_valid_inputs(tmp_path)
    (dataset / "cleaning_report").mkdir()
    (dataset / "cleaning_report/report.md").write_text("original report", encoding="utf-8")
    (dataset / "meta/modality.json").write_text("{}", encoding="utf-8")

    def fake_convert(repo_id, root, push_to_hub):
        assert push_to_hub is False
        staged = Path(root) / repo_id
        staged.rename(staged.with_name(repo_id + "_old"))
        (staged / "meta/episodes").mkdir(parents=True)
        (staged / "data").mkdir()
        (staged / "meta/info.json").write_text('{"codebase_version":"v3.0"}', encoding="utf-8")
        (staged / "meta/tasks.parquet").touch()

    module = types.ModuleType("lerobot.datasets.v30.convert_dataset_v21_to_v30")
    module.convert_dataset = fake_convert
    monkeypatch.setitem(sys.modules, module.__name__, module)
    export_lingbot_dataset(dataset, output)
    assert (output / "cleaning_report/report.md").read_text() == "original report"
    assert (output / "cleaning_report/source_modality.v21.json").is_file()
    assert json.loads((dataset / "meta/info.json").read_text())["codebase_version"] == "v2.1"


def test_converter_failure_does_not_publish_output(tmp_path, monkeypatch):
    import sys
    import types

    from lerobot_cleaner.v30.lingbot.vla import export_lingbot_dataset

    dataset, output, _ = make_valid_inputs(tmp_path)

    def fail(**kwargs):
        raise RuntimeError("conversion failed")

    module = types.ModuleType("lerobot.datasets.v30.convert_dataset_v21_to_v30")
    module.convert_dataset = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="conversion failed"):
        export_lingbot_dataset(dataset, output)
    assert not output.exists()
    assert (dataset / "meta/info.json").is_file()
