import pytest
from pydantic import ValidationError

from lerobot_cleaner.v21.config import CleaningConfig, OnMismatch


def test_defaults_and_reindex_enforced():
    cfg = CleaningConfig()
    assert cfg.rules.reindex_and_restats.enabled is True
    # try to disable -> validator forces it back on
    cfg2 = CleaningConfig(rules={"reindex_and_restats": {"enabled": False}})
    assert cfg2.rules.reindex_and_restats.enabled is True


def test_unknown_rule_key_rejected():
    with pytest.raises(ValidationError):
        CleaningConfig(rules={"not_a_rule": {"enabled": True}})


def test_unknown_field_in_rule_rejected():
    with pytest.raises(ValidationError):
        CleaningConfig(rules={"gripper_binarize": {"enabled": True, "bogus": 1}})


def test_yaml_roundtrip(tmp_path):
    cfg = CleaningConfig(
        input="a", output="b",
        rules={"timestamp_alignment": {"enabled": True, "on_mismatch": "strict_drop"}},
    )
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    loaded = CleaningConfig.from_yaml(p)
    assert loaded.rules.timestamp_alignment.enabled
    assert loaded.rules.timestamp_alignment.on_mismatch == OnMismatch.strict_drop


def test_preset_merge_user_wins(tmp_path):
    # preset sets threshold 0.5; user overrides to 0.9
    cfg = CleaningConfig(
        preset="galaxea_r1pro_dual_arm",
        rules={"gripper_binarize": {"threshold": 0.9}},
    )
    merged = cfg.merge_preset(
        {"recommended_rules": {"gripper_binarize": {"enabled": True, "threshold": 0.5,
                                                    "targets": ["state.left_gripper"]}}}
    )
    assert merged.rules.gripper_binarize.threshold == 0.9
    assert merged.rules.gripper_binarize.enabled is True
