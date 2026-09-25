"""Compatibility entry point for the packaged official LingBot norm runner."""
import sys
from lerobot_cleaner.v30.lingbot import norm as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
