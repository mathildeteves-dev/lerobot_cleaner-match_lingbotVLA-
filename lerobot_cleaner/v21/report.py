"""Cleaning report: markdown + json + optional matplotlib figures."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lerobot_cleaner.v21.config import CleaningConfig
    from lerobot_cleaner.v21.inspector import DatasetSummary
    from lerobot_cleaner.v21.parallel import EpisodeResult


@dataclass
class CleaningReport:
    config: "CleaningConfig"
    summary_before: "DatasetSummary"
    rule_names: list[str]
    summary_after: Optional["DatasetSummary"] = None

    dropped_episodes: list[tuple[int, str]] = field(default_factory=list)
    kept_episodes: int = 0
    rule_stats: dict[str, Counter] = field(default_factory=dict)
    notes: dict[int, dict[str, str]] = field(default_factory=dict)
    alignment_errors: list[str] = field(default_factory=list)
    _kept_lengths: list[int] = field(default_factory=list)
    episode_quality: dict[int, dict] = field(default_factory=dict)

    # --- accumulation -------------------------------------------------------
    def record_episode(self, res: "EpisodeResult") -> None:
        if res.dropped:
            self.dropped_episodes.append((res.src_index, res.drop_reason or "unknown"))
        else:
            self.kept_episodes += 1
            self._kept_lengths.append(res.length)
        self.episode_quality[res.src_index] = {
            "episode_index": res.src_index,
            "dropped": res.dropped,
            "drop_reason": res.drop_reason,
            "checks": {name: result.to_dict() for name, result in res.check_results.items()},
            "transforms": res.transform_results,
        }
        if res.notes:
            self.notes[res.src_index] = res.notes

    def merge_rule_stats(self, results: list["EpisodeResult"]) -> None:
        for res in results:
            for rule_name, counter in res.rule_stats.items():
                self.rule_stats.setdefault(rule_name, Counter()).update(counter)

    def add_alignment_error(self, err: str) -> None:
        self.alignment_errors.append(err)

    # --- output -------------------------------------------------------------
    def write(self, out_dir: Path, dry_run: bool = False) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # rule_stats may be empty in dry-run path; fold notes-only runs.
        if not self.rule_stats:
            self.rule_stats = {}
        self._write_episode_quality(out_dir / "episode_quality.jsonl")
        self._write_json(out_dir / "report.json", dry_run)
        self._write_markdown(out_dir / "report.md", dry_run)
        try:
            self._write_figures(out_dir / "figures")
        except Exception as e:  # matplotlib optional / headless issues
            (out_dir / "figures").mkdir(exist_ok=True)
            (out_dir / "figures" / "SKIPPED.txt").write_text(f"figures skipped: {e}\n")

    def _write_episode_quality(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for index in sorted(self.episode_quality):
                stream.write(json.dumps(self.episode_quality[index], ensure_ascii=False, allow_nan=False) + "\n")

    def _write_json(self, path: Path, dry_run: bool) -> None:
        data = {
            "dry_run": dry_run,
            "summary_before": self.summary_before.to_dict(),
            "summary_after": self.summary_after.to_dict() if self.summary_after else None,
            "kept_episodes": self.kept_episodes,
            "dropped_episodes": [
                {"episode_index": i, "reason": r} for i, r in self.dropped_episodes
            ],
            "rule_stats": {k: dict(v) for k, v in self.rule_stats.items()},
            "alignment_errors": self.alignment_errors,
            "notes": self.notes,
            "episode_modifications": [
                {"episode_index": index, "transforms": rec["transforms"]}
                for index, rec in sorted(self.episode_quality.items())
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_markdown(self, path: Path, dry_run: bool) -> None:
        b = self.summary_before
        a = self.summary_after
        lines = []
        title = "Cleaning Report (DRY RUN)" if dry_run else "Cleaning Report"
        lines.append(f"# {title}\n")
        lines.append(f"- **Input**: `{self.config.input}`")
        lines.append(f"- **Output**: `{self.config.output}`")
        lines.append(f"- **Enabled rules**: {', '.join(self.rule_names) or '(none)'}\n")

        lines.append("## Before / After\n")
        lines.append("| metric | before | after |")
        lines.append("|---|---|---|")

        def row(label, bv, av):
            lines.append(f"| {label} | {bv} | {av if av is not None else '—'} |")

        row("episodes", b.total_episodes, a.total_episodes if a else self.kept_episodes)
        row("frames", b.total_frames, a.total_frames if a else None)
        row("tasks", b.total_tasks, a.total_tasks if a else None)
        lines.append("")

        lines.append(f"**Kept**: {self.kept_episodes} episodes | "
                     f"**Dropped**: {len(self.dropped_episodes)} episodes\n")

        if self.alignment_errors:
            lines.append("## ⚠️ Alignment Errors\n")
            lines.append("These episodes failed the video↔parquet↔length invariant:\n")
            for e in self.alignment_errors:
                lines.append(f"- `{e}`")
            lines.append("")

        lines.append("## Transform Modifications\n")
        lines.append("| rule | episodes run | episodes changed |\n|---|---:|---:|")
        transform_counts = {}
        for record in self.episode_quality.values():
            for name, result in record["transforms"].items():
                counts = transform_counts.setdefault(name, [0, 0])
                counts[0] += 1
                counts[1] += int(result["changed"])
        for name, (runs, changed) in transform_counts.items():
            lines.append(f"| {name} | {runs} | {changed} |")
        lines.append("")

        lines.append("## Rule Trigger Counts\n")
        if self.rule_stats:
            for rule, counter in self.rule_stats.items():
                if not counter:
                    continue
                lines.append(f"### {rule}")
                for k, v in sorted(counter.items()):
                    lines.append(f"- {k}: {v}")
                lines.append("")
        else:
            lines.append("_No rule statistics recorded._\n")

        if self.dropped_episodes:
            lines.append("## Dropped Episodes\n")
            lines.append("| source episode | reason |")
            lines.append("|---|---|")
            for idx, reason in self.dropped_episodes:
                lines.append(f"| {idx} | {reason} |")
            lines.append("")

        path.write_text("\n".join(lines))

    def _write_figures(self, fig_dir: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir.mkdir(parents=True, exist_ok=True)

        if self._kept_lengths:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(self._kept_lengths, bins=min(30, max(5, len(self._kept_lengths))))
            ax.set_xlabel("episode length (frames)")
            ax.set_ylabel("count")
            ax.set_title("Kept episode length distribution")
            fig.tight_layout()
            fig.savefig(fig_dir / "episode_length_hist.png", dpi=110)
            plt.close(fig)

    # --- console summary ----------------------------------------------------
    def print_summary(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            t = Table(title="Cleaning Summary")
            t.add_column("metric")
            t.add_column("before", justify="right")
            t.add_column("after", justify="right")
            a = self.summary_after
            t.add_row("episodes", str(self.summary_before.total_episodes),
                      str(a.total_episodes if a else self.kept_episodes))
            t.add_row("frames", str(self.summary_before.total_frames),
                      str(a.total_frames if a else "—"))
            t.add_row("dropped", "—", str(len(self.dropped_episodes)))
            console.print(t)
            if self.alignment_errors:
                console.print(f"[red]⚠ {len(self.alignment_errors)} alignment errors[/red]")
        except Exception:
            print(f"Kept {self.kept_episodes}, dropped {len(self.dropped_episodes)}")
