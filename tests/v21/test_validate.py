"""Preflight input-contract validation tests."""

import json

import pytest

from lerobot_cleaner.v21.validate import InputContractError, validate_dataset


def test_valid_dataset_passes(synth_dataset):
    # The synthetic fixture is a well-formed GR00T v2.1 dataset.
    warnings = validate_dataset(synth_dataset)
    assert isinstance(warnings, list)


def test_missing_meta_dir(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(InputContractError, match="meta/"):
        validate_dataset(tmp_path)


def test_missing_modality_json(synth_dataset):
    (synth_dataset / "meta" / "modality.json").unlink()
    with pytest.raises(InputContractError, match="modality.json"):
        validate_dataset(synth_dataset)


def test_dim_mismatch_detected(synth_dataset):
    # Corrupt modality.json so declared state dim (5) != parquet width (4).
    mod_path = synth_dataset / "meta" / "modality.json"
    mod = json.loads(mod_path.read_text())
    mod["state"]["gripper"]["end"] = 5
    mod_path.write_text(json.dumps(mod))
    with pytest.raises(InputContractError, match="width"):
        validate_dataset(synth_dataset)


def test_nonexistent_root(tmp_path):
    with pytest.raises(InputContractError, match="does not exist"):
        validate_dataset(tmp_path / "nope")


def test_non_v21_version_warns(synth_dataset):
    info_path = synth_dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v2.0"
    info_path.write_text(json.dumps(info))
    warnings = validate_dataset(synth_dataset)
    assert any("v2.0" in w for w in warnings)
