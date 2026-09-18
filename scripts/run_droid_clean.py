"""Audit/clean DROID with bounded batches and paths independent of terminal location."""

import argparse
import json
from pathlib import Path

from lerobot_cleaner.v30.lingbot.config import validate_mapping
from lerobot_cleaner.v30.v3 import V3Config, audit_v3, clean_v3

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT.parent / "数据实例/droid_100_lerobotv3/droid_100_lerobotv3"
DEFAULT_OUTPUT = PROJECT.parent / "清洗结果/droid_100_clean_v3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/cleaning/droid_v3.yaml")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-videos", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed numeric phase and verified video copies",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress bars")
    args = parser.parse_args()
    config = V3Config.from_yaml(args.config)
    config.verify_videos = config.verify_videos or args.verify_videos
    config.progress = config.progress and not args.quiet
    if args.audit_only and args.resume:
        parser.error("--resume applies to cleaning, not read-only audit")
    mapping = validate_mapping(
        args.dataset,
        PROJECT / "configs/robot_configs/droid_franka.yaml",
        PROJECT / "configs/vla/droid_franka.yaml",
    )
    if args.audit_only:
        audit = audit_v3(args.dataset, config)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in audit.items()
                    if k not in {"numeric", "videos", "data_file_selection"}
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if "data_file_selection" in audit:
            selection = audit["data_file_selection"]
            selected = selection["selected_files"]
            print(f"Data file policy: {selection['policy']}")
            print(
                f"Selected files: {len(selected)} / {selection['discovered_files']}; "
                f"excluded files: {len(selection['excluded_files'])}; "
                f"excluded rows (readable footers): {selection['excluded_rows_known']}"
            )
            print(f"Selected first/last: {selected[:1]} / {selected[-1:]}")
        nonfinite = {k: v["nonfinite"] for k, v in audit["numeric"].items() if v["nonfinite"]}
        print(f"Non-finite values by feature: {nonfinite}")
        print(json.dumps(mapping, indent=2))
        return
    report = clean_v3(args.dataset, args.output, config, resume=args.resume)
    report_dir = args.output / "cleaning_report"
    (report_dir / "lingbot_mapping_validation.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    print(f"Output: {args.output.resolve()}")
    print(f"Changed values: {report['changed_values']}; removed rows: {report['rows_removed']}")
    print(f"Report: {report_dir.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
