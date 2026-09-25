"""Portable automatic reports; completion is published only after this succeeds."""

import hashlib
import html
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lerobot_cleaner.v30.v3_streaming import digest, write_json


def provenance(project):
    project = Path(project)
    versions = {}
    for name in ["numpy", "pandas", "pyarrow", "av", "pydantic"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    code_hash = hashlib.sha256()
    for folder in ["lerobot_cleaner", "scripts"]:
        for path in sorted((project / folder).rglob("*.py")):
            code_hash.update(path.relative_to(project).as_posix().encode())
            code_hash.update(path.read_bytes())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "python": platform.python_version(),
        "packages": versions,
        "git_commit": commit,
        "working_code_sha256": code_hash.hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def report_status(report):
    numeric = report.get("numeric_after", report.get("numeric", {}))
    failures = {k: v["nonfinite"] for k, v in numeric.items() if v["nonfinite"]}
    candidates = report.get("output_review", report.get("dataset_review", {})).get(
        "quality_candidates", 0
    )
    visual = sum(
        bool(row["flags"])
        for video in report.get("videos", {}).values()
        for row in video.get("visual_review", {}).get("episodes", [])
    )
    language = report.get("dataset_quality", {}).get("language", {})
    return {
        "language_integrity_failed_episodes": language.get("episodes_failed"),
        "language_warning_count": language.get("warning_count"),
        "status": "failed"
        if failures or language.get("episodes_failed", 0)
        else (
            "warning"
            if candidates or visual or language.get("warning_count", 0) or report.get("video_verification") != "full_decode"
            else "passed"
        ),
        "nonfinite": failures,
        "numeric_quality_candidate_episodes": candidates,
        "visual_candidate_camera_episodes": visual,
        "video_verification": report.get("video_verification"),
        "scope": "data integrity and heuristic quality review; not task success or training approval",
    }


def write_review_bundle(folder, report, profile, config, *, project, training=None, source=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    review = report.get("output_review", report.get("dataset_review", {}))
    # Temporary staging paths must not escape into published provenance.
    review = dict(review)
    review["input"] = report.get("input", report.get("dataset"))
    summary = report_status(report)
    training = training or {
        "status": "blocked",
        "runtime_validated": False,
        "blockers": ["Action semantics not verified", "Actual LingBot batch not loaded"],
    }
    report["review_summary"] = summary
    # Runtime/semantic preflight is not the model contract result. Preserve both.
    report["training_runtime_preflight"] = training
    training = report.get("training_readiness", training)
    report["training_readiness"] = training
    report["provenance"] = provenance(project)
    for key in ["output_review"]:
        if key in report:
            report[key]["input"] = report.get("output", report[key]["input"])
    write_json(folder / "report.json", report)
    write_json(folder / "quality_summary.json", summary)
    write_json(folder / "episode_quality.json", review.get("episode_quality", []))
    write_json(
        folder / "libero_validation.json",
        {k: v for k, v in review.items() if k != "episode_quality"},
    )
    write_json(folder / "training_readiness.json", training)
    write_json(folder / "language_quality.json", report.get("dataset_quality", {}).get("language", {"status": "not_evaluated"}))
    (folder / "profile.used.yaml").write_text(
        yaml.safe_dump(profile.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (folder / "cleaning_config.used.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8"
    )
    if source:
        stage = folder.parent
        meta_checks = {
            name: digest(Path(source) / "meta" / name) == digest(stage / "meta" / name)
            for name in ["info.json", "tasks.parquet"]
        }
        if not all(meta_checks.values()):
            raise ValueError("Output info/tasks differ from source")
        write_json(
            folder / "post_validation.json",
            {
                "metadata_byte_identical": meta_checks,
                "output_rescanned": True,
                "numeric_before_after_equal": report["numeric_before"] == report["numeric_after"],
                "meaning": "Numeric summaries compared; no claim of full cell equality from summary alone",
                "video_files_verified": len(report.get("video_sha256", {})),
                "output_profile_revalidated": "output_review" in report,
            },
        )
    visual_rows = []
    for path, item in report.get("videos", {}).items():
        for row in item.get("visual_review", {}).get("episodes", []):
            visual_rows.append({"video": path, "camera": item["key"], **row})
    write_json(folder / "video_quality.json", visual_rows)
    labels = {"passed": "通过", "warning": "通过完整性检查，存在待复核项", "failed": "失败"}
    lines = [
        "# 数据清洗与质量检查报告",
        "",
        f"检查结论：**{labels[summary['status']]}**。",
        "",
        f"输入：`{review.get('input', '')}`",
        f"输出：`{report.get('output', '只检查，不生成数据')}`",
        "",
        f"轨迹：{report['episodes']}；帧：{report['frames']}；任务：{report['tasks']}；FPS：{report['fps']}。",
        "",
        f"非有限数值：{summary['nonfinite'] or '未发现'}。",
        f"语言完整性异常轨迹：{summary['language_integrity_failed_episodes']}；语言提示：{summary['language_warning_count']}。详见 language_quality.json。",
        f"数值质量候选轨迹：{summary['numeric_quality_candidate_episodes']}；视频候选（相机×轨迹）：{summary['visual_candidate_camera_episodes']}。",
        f"视频验证方式：{report['video_verification']}；修改数值：{report.get('changed_values', '仅检查')}；删除帧：{report.get('rows_removed', '仅检查')}。",
        "",
        "质量阈值是原始数值单位下的启发式提示，不代表轨迹失败；静止可能是正常操作，画面低细节不等同于模糊。不会自动删除候选轨迹。",
        "",
        "成功/失败数量："
        + (
            str(report.get("unsuccessful_episodes")) + " 条失败（源标签）"
            if review.get("success_labels_available")
            else "未知，源数据无成功标签。"
        ),
        "",
        "## 任务分布",
        "",
        "| 任务 | 轨迹 | 帧 | 文本 |",
        "|---|---:|---:|---|",
    ]
    for row in review.get("tasks", []):
        lines.append(
            f"| {row['task_index']} | {row['episodes']} | {row['frames']} | {row['text'].replace('|', '/').replace(chr(10), ' ')} |"
        )
    lines += ["", "## 视频验证", "", "| 文件 | 解码帧数 | 使用检查点 |", "|---|---:|---|"]
    for path, item in report.get("videos", {}).items():
        lines.append(
            f"| {path} | {item.get('decoded_frames', '未解码')} | {item.get('decode_reused', False)} |"
        )
    lines += ["", "## 训练就绪", "", f"状态：{training['status']}。", ""]
    lines += [f"- {reason}" for reason in training.get("blockers", [])]
    lines += [f"- {item['severity']}: {item['message']}" for item in training.get("findings", [])]
    lines += [
        "",
        "## 复核入口",
        "",
        "打开 review.html 查看每条轨迹的候选原因及按任务抽取的中间帧。完整指标见 episode_quality.json 和 video_quality.json。",
        "抽样画面不能代替完整任务视频复核。数值统计重新计算，图像统计沿用输入。",
        "",
        "报告记录代码哈希、软件版本、实际配置和视频校验信息。COMPLETE.json 仅在所有输出复核和报告生成成功后出现。",
    ]
    (folder / "analysis_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    visual_by_episode = {}
    for row in visual_rows:
        visual_by_episode.setdefault(row["episode_index"], []).append(row)
    rows = []
    for row in review.get("episode_quality", []):
        visual = visual_by_episode.get(row["episode_index"], [])
        flags = row["flags"] + [f"{v['camera']}: {flag}" for v in visual for flag in v["flags"]]
        previews = "".join(
            f'<img width="192" loading="lazy" src="{html.escape(v["preview"].removeprefix("cleaning_report/"), quote=True)}" alt="episode {row["episode_index"]}">'
            for v in visual
            if v.get("preview")
        )
        rows.append(
            f'<tr data-flags="{int(bool(flags))}"><td>{row["episode_index"]}</td><td>{html.escape(row["task"])}</td><td>{row["seconds"]:.2f}</td><td>{html.escape(", ".join(flags) or "未标记")}</td><td>{previews}</td></tr>'
        )
    page = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>轨迹复核</title><style>body{font-family:system-ui;margin:28px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccc;padding:10px;text-align:left}img{margin:4px}input{margin:12px}</style><h1>轨迹质量复核</h1><p>启发式标记，不自动删除。图像为部分轨迹中间帧；完整任务仍需人工观看视频。</p><label><input type="checkbox" onchange="document.querySelectorAll(\'tr[data-flags="0"]\').forEach(r=>r.hidden=this.checked)">只显示候选轨迹</label><table><thead><tr><th>轨迹</th><th>任务</th><th>秒</th><th>候选原因</th><th>抽查画面</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></html>"
    )
    (folder / "review.html").write_text(page, encoding="utf-8")
    return summary
