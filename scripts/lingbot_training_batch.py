"""Compatibility import for the packaged LingBot preprocessing/collator smoke."""
import sys
from lerobot_cleaner.training.smoke import batch as _implementation
sys.modules[__name__] = _implementation
