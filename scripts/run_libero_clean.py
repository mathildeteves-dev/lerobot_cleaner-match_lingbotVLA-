"""Profile-based LIBERO review and atomic cleaning with automatic reports."""

import argparse
import json
from pathlib import Path

from lerobot_cleaner.v30.episode_review import DatasetReview
from lerobot_cleaner.v30.review_profile import load_profile
from lerobot_cleaner.v30.review_report import provenance, report_status, write_review_bundle
from lerobot_cleaner.v30.training_readiness import check_readiness
from lerobot_cleaner.v30.v3 import V3Config
from lerobot_cleaner.v30.v3_streaming import (
    audit_streaming,
    clean_streaming,
    job_lock,
    verify_videos,
    write_json,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT.parent / "数据实例/lerobot_v30/libero_10_no_noops_lerobot"
DEFAULT_OUTPUT = PROJECT.parent / "清洗结果/libero_10_clean_v3_reviewed"


def run(args):
    config = V3Config.from_yaml(args.config)
    if config.engine != "streaming":
        raise ValueError("LIBERO entry point requires engine=streaming")
    config.progress = config.progress and not args.quiet
    if args.quick:
        config.verify_videos = False
    profile = load_profile(args.profile)
    output, source = args.output.resolve(), args.dataset.resolve()
    if (
        output.exists()
        or source == output
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("Output must be new and outside the input dataset")
    print(f"Input: {source}\nOutput: {output}\nProfile: {profile.name}", flush=True)

    def factory(root):
        return DatasetReview(root, profile)

    training = check_readiness(profile, args.lingbot_root)
    settings = {
        "profile": profile.model_dump(mode="json"),
        "working_code_sha256": provenance(PROJECT)["working_code_sha256"],
        "lingbot_root": str(args.lingbot_root.resolve()) if args.lingbot_root else None,
    }

    def finalize(stage, report):
        write_review_bundle(
            stage / "cleaning_report",
            report,
            profile,
            config,
            project=PROJECT,
            training=training,
            source=source,
        )

    if args.audit_only:
        if args.resume:
            raise ValueError("Audit has no dataset-copy resume; omit --resume")
        partial = output.with_name(output.name + ".partial")
        partial.mkdir(parents=True, exist_ok=False)
        with job_lock(partial):
            report = audit_streaming(
                source, config.model_copy(update={"verify_videos": False}), observer=factory(source)
            )
            if config.verify_videos:
                visual = profile.quality.visual.model_dump(mode="json")
                if visual.pop("enabled"):
                    seen, selected = set(), []
                    limit = visual.pop("preview_limit")
                    for row in report["dataset_review"]["episode_quality"]:
                        if row["task_index"] not in seen and len(selected) < limit:
                            selected.append(row["episode_index"])
                            seen.add(row["task_index"])
                    visual["preview_episodes"], visual["preview_prefix"] = selected, ""
                else:
                    visual = None
                info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
                verify_videos(
                    source, info, report["videos"], config, quality=visual, preview_root=partial
                )
                report["video_verification"] = "full_decode"
            write_review_bundle(
                partial, report, profile, config, project=PROJECT, training=training
            )
            write_json(partial / "AUDIT_COMPLETE.json", {"status": report_status(report)["status"]})
        if output.exists():
            raise ValueError("Output appeared during audit")
        partial.rename(output)
    else:
        visual = profile.quality.visual.model_dump(mode="json")
        visual = visual if visual.pop("enabled") else None
        report = clean_streaming(
            source,
            output,
            config,
            resume=args.resume,
            observer_factory=factory,
            finalize=finalize,
            job_metadata=settings,
            video_quality=visual,
        )
    summary = report_status(report)
    print(
        json.dumps(
            {
                "episodes": report["episodes"],
                "frames": report["frames"],
                "tasks": report["tasks"],
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Training readiness: " + training["status"])
    print(
        f"Report: {output / ('analysis_zh.md' if args.audit_only else 'cleaning_report/analysis_zh.md')}"
    )
    return 2 if summary["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/cleaning/libero_v3.yaml")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip video decoding; explicitly reported as incomplete video verification",
    )
    parser.add_argument("--lingbot-root", type=Path)
    args = parser.parse_args()
    args.output = args.output or (
        DEFAULT_OUTPUT.with_name("libero_10_audit") if args.audit_only else DEFAULT_OUTPUT
    )
    try:
        return run(args)
    except Exception as exc:
        output, source = args.output.resolve(), args.dataset.resolve()
        if (
            output != source
            and not output.is_relative_to(source)
            and not source.is_relative_to(output)
        ):
            failure = output.with_name(output.name + ".failure.json")
            if not failure.exists():
                failure.parent.mkdir(parents=True, exist_ok=True)
                with failure.open("x", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "status": "failed",
                            "error": str(exc),
                            "input": str(source),
                            "output": str(output),
                            "final_output_created": output.exists(),
                        },
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"Failure report: {failure}")
        print(f"FAILED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
