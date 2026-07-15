"""R6: numeric sanity — NaN/Inf, joint limits, and percentile outlier handling.

Outlier handling matters more than it looks: GR00T's default normalization is
min/max (use_mean_std=False), so a single extreme frame stretches the normalized
range across the whole dataset. Clipping by per-dataset quantiles is therefore a
high-value cleaning step. The actual clip bounds are computed in a cheap
pre-pass over the source (see _compute_outlier_bounds in the pipeline) and passed
in via config; here we apply them per episode.
"""

from __future__ import annotations

import numpy as np

from lerobot_cleaner.config import OnBad, OutlierMode
from lerobot_cleaner.dataset.reader import ACTION_COL, STATE_COL
from lerobot_cleaner.rules.base import Rule
from lerobot_cleaner.types import EpisodeWork


def _stack(df, col):
    return np.stack(df[col].to_numpy()) if col in df.columns and len(df) else None


def _unstack_into(df, col, arr):
    df[col] = list(arr)


class NumericSanityRule(Rule):
    name = "numeric_sanity"

    def __init__(self, config, dataset, outlier_bounds=None):
        super().__init__(config, dataset)
        # outlier_bounds: {"state": (low_vec, high_vec), "action": (...)}
        self.outlier_bounds = outlier_bounds or {}

    def apply(self, work: EpisodeWork) -> None:
        df = work.df
        for col, modality in ((STATE_COL, "state"), (ACTION_COL, "action")):
            arr = _stack(df, col)
            if arr is None:
                continue
            arr = arr.astype(np.float64)

            # --- NaN / Inf ---
            nan_mask = np.isnan(arr).any(axis=1)
            inf_mask = np.isinf(arr).any(axis=1)
            if nan_mask.any():
                if not self._handle_bad(work, col, arr, nan_mask, self.config.on_nan, "nan"):
                    return
                arr = _stack(work.df, col).astype(np.float64)
                df = work.df
            if inf_mask.any():
                if not self._handle_bad(work, col, arr, inf_mask, self.config.on_inf, "inf"):
                    return
                arr = _stack(work.df, col).astype(np.float64)
                df = work.df

            # --- joint limits ---
            if self.config.joint_limits:
                arr = self._apply_joint_limits(work, col, modality, arr)

            # --- outlier clipping ---
            if self.config.outlier_mode != OutlierMode.off and modality in self.outlier_bounds:
                arr = self._apply_outliers(work, col, modality, arr)

            _unstack_into(work.df, col, arr)

    # --- helpers ------------------------------------------------------------
    def _handle_bad(self, work, col, arr, bad_mask, action: OnBad, label: str) -> bool:
        """Returns True to continue processing this episode, False if dropped."""
        n_bad = int(bad_mask.sum())
        self.stats[f"{label}_frames_detected"] += n_bad
        if action == OnBad.drop_episode:
            work.drop(f"{label} in {col}")
            self.stats[f"episodes_dropped_{label}"] += 1
            return False
        if action == OnBad.warn:
            work.note(self.name, f"{n_bad} {label} frames in {col} (warn only)")
            return True
        if action == OnBad.drop_frame:
            work.restrict_to(~bad_mask)
            self.stats[f"{label}_frames_dropped"] += n_bad
            return True
        if action == OnBad.interpolate:
            self._interpolate(work, col, bad_mask)
            self.stats[f"{label}_frames_interpolated"] += n_bad
            return True
        if action == OnBad.clip:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            _unstack_into(work.df, col, arr)
            self.stats[f"{label}_frames_clipped"] += n_bad
            return True
        return True

    def _interpolate(self, work, col, bad_mask) -> None:
        arr = _stack(work.df, col).astype(np.float64)
        arr[bad_mask] = np.nan
        n, d = arr.shape
        x = np.arange(n)
        for j in range(d):
            colj = arr[:, j]
            good = ~np.isnan(colj)
            if good.sum() == 0:
                arr[:, j] = 0.0
            elif good.sum() < n:
                arr[:, j] = np.interp(x, x[good], colj[good])
        _unstack_into(work.df, col, arr)

    def _apply_joint_limits(self, work, col, modality, arr):
        for dotted, (low, high) in self.config.joint_limits.items():
            mod, key = dotted.split(".", 1)
            if mod != modality:
                continue
            sl = self.resolver.resolve(dotted)
            sub = arr[:, sl.start : sl.end]
            violated = ((sub < low) | (sub > high)).any(axis=1)
            n_viol = int(violated.sum())
            if n_viol == 0:
                continue
            self.stats["limit_violations"] += n_viol
            if self.config.on_limit_violation == OnBad.clip:
                arr[:, sl.start : sl.end] = np.clip(sub, low, high)
            elif self.config.on_limit_violation == OnBad.drop_frame:
                work.restrict_to(~violated)
                arr = _stack(work.df, col).astype(np.float64)
            elif self.config.on_limit_violation == OnBad.drop_episode:
                work.drop(f"joint limit violation in {dotted}")
        return arr

    def _apply_outliers(self, work, col, modality, arr):
        low_vec, high_vec = self.outlier_bounds[modality]
        targets = self.config.outlier_targets
        if targets:
            # Build a column mask restricted to target keys of this modality.
            mask = np.zeros(arr.shape[1], dtype=bool)
            for dotted in targets:
                mod, key = dotted.split(".", 1)
                if mod == modality:
                    sl = self.resolver.resolve(dotted)
                    mask[sl.start : sl.end] = True
        else:
            mask = np.ones(arr.shape[1], dtype=bool)

        out_of_range = ((arr < low_vec) | (arr > high_vec)) & mask
        rows_with_outliers = out_of_range.any(axis=1)
        n = int(rows_with_outliers.sum())
        if n == 0:
            return arr
        self.stats[f"outlier_frames_{modality}"] += n
        if self.config.outlier_mode == OutlierMode.warn:
            work.note(self.name, f"{n} outlier frames in {col} (warn only)")
        elif self.config.outlier_mode == OutlierMode.clip_quantile:
            clipped = np.where(mask, np.clip(arr, low_vec, high_vec), arr)
            arr = clipped
            self.stats[f"outlier_frames_clipped_{modality}"] += n
        elif self.config.outlier_mode == OutlierMode.drop_frame:
            work.restrict_to(~rows_with_outliers)
            arr = _stack(work.df, col).astype(np.float64)
        return arr
