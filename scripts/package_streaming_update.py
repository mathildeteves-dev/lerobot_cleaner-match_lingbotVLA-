"""Build a small source-only ZIP for updating an existing server checkout."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT = Path(__file__).resolve().parents[1]
FILES = [
    "lerobot_cleaner/__init__.py",
    "lerobot_cleaner/v30/__init__.py",
    "lerobot_cleaner/v30/v3.py",
    "lerobot_cleaner/v30/v3_streaming.py",
    "lerobot_cleaner/v30/v3_stream_stats.py",
    "lerobot_cleaner/v30/lingbot/__init__.py",
    "lerobot_cleaner/v30/lingbot/config.py",
    "lerobot_cleaner/v30/lingbot/generate.py",
    "lerobot_cleaner/v30/lingbot/vla.py",
    "lerobot_cleaner/cli.py",
    "scripts/run_droid_clean.py",
    "configs/cleaning/droid_v3.yaml",
    "configs/cleaning/droid_v3_referenced.yaml",
    "tests/v30/test_v3.py",
    "tests/v30/test_v3_streaming.py",
    "docs/STREAMING_SERVER_ZH.md",
    "docs/STALE_SHARDS_ZH.md",
    "docs/DROID_GUIDE_ZH.md",
    "README.md",
    "PROJECT_STATUS.md",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "dist/streaming-update.zip")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output archive exists; choose a different --output filename")
    for name in FILES:
        if not (PROJECT / name).is_file():
            parser.error(f"Missing source: {name}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in FILES}
    with ZipFile(args.output, "x", ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(PROJECT / name, name)
        archive.writestr("STREAMING_UPDATE_MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"Source update: {args.output.resolve()} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
