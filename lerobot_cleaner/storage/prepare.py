"""Normalize versions using the official converter, always on a separate copy."""
import hashlib
import json
from pathlib import Path


def _identity(root):
    digest = hashlib.sha256()
    for directory in ("meta", "data", "videos"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                stat = path.stat()
                digest.update(f"{path.relative_to(root).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
                if directory == "meta":
                    digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_dataset(root, config=None):
    """v3 is passed through; v2.1 requires an explicit persistent conversion path.

    This minimal version probe precedes LeRobotDataset because 0.4.2 cannot open
    v2.1. It does not interpret episode offsets or robot feature semantics.
    """
    root = Path(root).resolve()
    version = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))["codebase_version"]
    if version == "v3.0":
        return root
    if version != "v2.1":
        raise ValueError(f"Unsupported LeRobot version: {version}")
    destination = getattr(config, "converted_root", None)
    if destination is None:
        raise ValueError("v2.1 input requires converted_root in the config, or export-lingbot first")
    destination = Path(destination).resolve()
    if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("converted_root must be outside the source dataset")
    marker = destination / "cleaning_report/source_conversion.json"
    identity = {"source": str(root), "fingerprint": _identity(root), "converter": "lerobot-v0.4.2"}
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity:
            raise ValueError("Conversion destination is occupied or source changed; choose a new converted_root")
        converted_info = json.loads((destination / "meta/info.json").read_text(encoding="utf-8"))
        if converted_info.get("codebase_version") != "v3.0":
            raise ValueError("Cached conversion is not v3.0")
        return destination
    from lerobot_cleaner.v30.lingbot.vla import export_lingbot_dataset
    export_lingbot_dataset(root, destination)
    if _identity(root) != identity["fingerprint"]:
        raise ValueError("Source changed during official conversion; converted copy is not reusable")
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return destination
