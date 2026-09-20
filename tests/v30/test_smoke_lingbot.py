"""Real LingBot integration gate; never substitutes a fake dataset or runtime."""
import os
from pathlib import Path

import pytest

from scripts.smoke_lingbot import smoke


def test_real_lingbot_smoke():
    required = ["LINGBOT_SMOKE_DATASET", "LINGBOT_SMOKE_OUTPUT", "LINGBOT_ROOT",
                "LINGBOT_SMOKE_PROFILE", "LINGBOT_ROBOT_CONFIG", "LINGBOT_TRAIN_CONFIG"]
    if not all(os.environ.get(key) for key in required):
        pytest.skip("Real LingBot environment/data paths not configured; smoke has NOT passed")
    paths = [Path(os.environ[key]) for key in required]
    level = int(os.environ.get("LINGBOT_SMOKE_LEVEL", "1"))
    result = smoke(*paths, level=level, cleaning_config=Path(os.environ["LINGBOT_CLEANING_CONFIG"]) if os.environ.get("LINGBOT_CLEANING_CONFIG") else None)
    assert result["status"] == "passed", result
    expected = [
        "clean-v3", "validate-lingbot", "compute_norm.py", "VLADataset", "dataset[0]", "DataLoader batch"]
    if level >= 2:
        expected += ["processor/tokenizer", "FeatureTransform(do_nomalize=True)",
                     "pad_and_concat", "prepare_images", "prepare_language", "training-shaped batch"]
        assert result["loader_report"]["training_batch_validated"] is True
    if level == 3:
        expected += ["model forward"]
        assert result["model_forward_validated"] is True
    assert result["completed_level"] == level
    assert [step["stage"] for step in result["stages"]] == expected
