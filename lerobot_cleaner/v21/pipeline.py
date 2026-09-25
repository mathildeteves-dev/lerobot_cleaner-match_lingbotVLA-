"""Cleaning pipeline orchestration.

Flow:
  1. (optional) outlier pre-pass: sample source state/action to get q01/q99 bounds
  2. parallel per-episode: checks -> accept/reject -> transforms -> stage artifacts
  3. sequential finalize: assign new indices, rebuild index cols + uniform ts,
     move videos into final layout, accumulate exact streaming stats
  4. verify alignment (video frames == rows == episodes.jsonl length)
  5. write all meta (R8)
  6. emit report
"""

from __future__ import annotations

import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lerobot_cleaner.v21.legacy_reader import ACTION_COL, STATE_COL, LegacyGrootAdapter as GrootAdapter
from lerobot_cleaner.v21.config import CleaningConfig, OutlierMode
from lerobot_cleaner.v21.inspector import inspect_dataset
from lerobot_cleaner.v21.parallel import EpisodeResult, process_episode
from lerobot_cleaner.v21.report import CleaningReport
from lerobot_cleaner.v21.rules import (
    build_checks,
    build_finalizers,
    build_transforms,
    run_episode_stages,
)
from lerobot_cleaner.v21.rules.checks.trajectory._common import finite_quantile_bounds
from lerobot_cleaner.v21.validate import validate_dataset
from lerobot_cleaner.v21.writer import DatasetWriter


class Pipeline:
    def __init__(self, config: CleaningConfig):
        if config.input is None or config.output is None:
            raise ValueError("config.input and config.output must be set")
        self.config = config
        self.input = Path(config.input)
        self.output = Path(config.output)
        if self.input.resolve() == self.output.resolve():
            raise ValueError("output must differ from input (never mutate source)")
        # Preflight: fail fast with one clear error if the input does not meet
        # the GR00T-format LeRobot v2.1 contract. Warnings are surfaced via run().
        self.input_warnings = validate_dataset(self.input)
        self.source = GrootAdapter(self.input)

    # --- outlier pre-pass ---------------------------------------------------
    def _compute_outlier_bounds(self, refs) -> dict:
        nsc = self.config.rules.numeric_sanity
        if nsc.outlier_mode in (OutlierMode.off, OutlierMode.warn) and not nsc.enabled:
            return {}
        if nsc.outlier_mode == OutlierMode.off:
            return {}
        # Sample up to ~200k rows total across episodes for quantile bounds.
        max_rows = 200_000
        per_ep = max(1, max_rows // max(1, len(refs)))
        state_chunks, action_chunks = [], []
        for ref in refs:
            try:
                df = self.source.read_episode(ref.episode_index).df
            except Exception:
                continue
            if STATE_COL in df.columns and len(df):
                arr = np.stack(df[STATE_COL].to_numpy()).astype(np.float64)
                state_chunks.append(arr[:: max(1, len(arr) // per_ep + 1)])
            if ACTION_COL in df.columns and len(df):
                arr = np.stack(df[ACTION_COL].to_numpy()).astype(np.float64)
                action_chunks.append(arr[:: max(1, len(arr) // per_ep + 1)])
        bounds = {}
        if state_chunks:
            s = np.concatenate(state_chunks)
            bounds["state"] = finite_quantile_bounds(
                s, nsc.outlier_low_quantile, nsc.outlier_high_quantile
            )
        if action_chunks:
            a = np.concatenate(action_chunks)
            bounds["action"] = finite_quantile_bounds(
                a, nsc.outlier_low_quantile, nsc.outlier_high_quantile
            )
        return bounds

    # --- main run -----------------------------------------------------------
    def run(self) -> CleaningReport:
        summary_before = inspect_dataset(self.source)
        refs = self.source.episodes()

        outlier_bounds = {}
        if self.config.rules.numeric_sanity.enabled:
            outlier_bounds = self._compute_outlier_bounds(refs)

        checks = build_checks(self.config, self.source, outlier_bounds=outlier_bounds)
        transforms = build_transforms(self.config, self.source, outlier_bounds=outlier_bounds)
        finalizers = build_finalizers(self.config)

        report = CleaningReport(
            config=self.config,
            summary_before=summary_before,
            rule_names=[r.name for r in checks + transforms]
            + ([] if self.config.dry_run else [r.name for r in finalizers]),
        )

        if self.config.dry_run:
            return self._dry_run(refs, checks, transforms, report)

        # Output indices cannot identify which source episodes were already processed.
        # Until source-to-output provenance supports resume, fail before writing anything.
        if self.config.resume:
            raise ValueError("v2.1 resume is unsupported after reindexing; use a new output directory")
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError("output directory must be empty; use a new output directory")
        self.output.mkdir(parents=True, exist_ok=True)
        writer = DatasetWriter(self.source, self.output, codec="libx264")

        staging_root = Path(tempfile.mkdtemp(prefix="lerobot_clean_", dir=self.output))
        codec = "libx264"
        fps = self.source.fps
        video_keys = self.source.resolver.video_keys()

        results: list[EpisodeResult] = []
        try:
            todo = refs
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as ex:
                futures = {
                    ex.submit(
                        process_episode, ref, checks, transforms, staging_root, codec, fps, video_keys, self.source
                    ): ref
                    for ref in todo
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc="cleaning"):
                    results.append(fut.result())

            for finalizer in finalizers:
                finalizer.finalize(results, writer, report)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

        report.summary_after = inspect_dataset(GrootAdapter(self.output))
        report.merge_rule_stats(results)
        self._write_used_config()
        report.write(self.output / "cleaning_report")
        return report

    # --- dry run ------------------------------------------------------------
    def _dry_run(self, refs, checks, transforms, report: CleaningReport) -> CleaningReport:
        from lerobot_cleaner.v21.parallel import EpisodeResult

        results = []
        for ref in tqdm(refs, desc="dry-run"):
            work = self.source.read_episode(ref.episode_index)
            res_stats = run_episode_stages(work, checks, transforms)
            res = EpisodeResult(
                src_index=ref.episode_index,
                dropped=work.dropped,
                drop_reason=work.drop_reason,
                length=len(work.df),
                tasks=ref.tasks,
                staged_parquet=None,
                staged_videos={},
                notes=work.notes,
                rule_stats=res_stats,
                check_results=work.check_results,
                transform_results=work.transform_results,
            )
            report.record_episode(res)
            results.append(res)
        report.merge_rule_stats(results)
        report.write(self.output / "cleaning_report", dry_run=True)
        return report

    def _write_used_config(self) -> None:
        out = self.output / "cleaning_report"
        out.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(out / "cleaning_config.used.yaml")
