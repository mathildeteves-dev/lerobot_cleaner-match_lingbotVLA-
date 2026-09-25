"""Sampled visual diagnostics collected during the existing full decode pass."""

import hashlib

import numpy as np


class VideoReview:
    def __init__(self, root, relative, item, info, options):
        self.root, self.item, self.options = root, item, options
        self.fps = info["fps"]
        self.stride = max(1, round(self.fps * options["sample_seconds"]))
        self.prefix = hashlib.sha256(relative.encode()).hexdigest()[:12]
        self.rows = {}
        self.previous = None
        self.previous_interval = None
        self.preview_paths = []

    def sample(self, frame, interval, timestamp, count):
        episode = self.item["episode_indices"][interval]
        row = self.rows.setdefault(
            interval,
            {
                "episode_index": episode,
                "samples": 0,
                "black_samples": 0,
                "low_detail_samples": 0,
                "unchanged_transitions": 0,
                "longest_unchanged_sample_span_seconds": 0.0,
                "preview": None,
            },
        )
        start, end = self.item["intervals"][interval]
        if (
            episode in self.options.get("preview_episodes", [])
            and row["preview"] is None
            and timestamp >= (start + end) / 2
        ):
            prefix = self.options.get("preview_prefix", "cleaning_report/")
            name = f"{prefix}previews/{self.prefix}_{episode}.jpg"
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_image().resize((256, 256)).save(path, quality=80)
            row["preview"] = name
            self.preview_paths.append(name)
        if (count - 1) % self.stride:
            return
        gray = frame.to_ndarray(format="gray")
        gray = gray[:: max(1, gray.shape[0] // 64), :: max(1, gray.shape[1] // 64)].astype(float)
        row["samples"] += 1
        row["black_samples"] += int(np.mean(gray <= self.options["black_level"]) >= 0.98)
        lap = (
            gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
            - 4 * gray[1:-1, 1:-1]
        )
        row["low_detail_samples"] += int(lap.var() < self.options["low_detail_variance"])
        if (
            self.previous_interval == interval
            and np.mean(np.abs(gray - self.previous)) <= self.options["unchanged_pixel_tolerance"]
        ):
            row["unchanged_transitions"] += 1
            row["_run"] = row.get("_run", 0) + 1
            row["longest_unchanged_sample_span_seconds"] = max(
                row["longest_unchanged_sample_span_seconds"], row["_run"] * self.stride / self.fps
            )
        else:
            row["_run"] = 0
        self.previous, self.previous_interval = gray, interval

    def result(self):
        rows = []
        for row in self.rows.values():
            row = {k: v for k, v in row.items() if not k.startswith("_")}
            row["flags"] = []
            if row["black_samples"]:
                row["flags"].append("sampled_black_frames")
            if row["samples"] and row["low_detail_samples"] / row["samples"] >= 0.5:
                row["flags"].append("sampled_low_detail")
            if row["longest_unchanged_sample_span_seconds"] >= self.options["unchanged_seconds"]:
                row["flags"].append("sampled_unchanged_scene")
            rows.append(row)
        return {
            "method": "periodic grayscale sampling during full decode",
            "sample_seconds": self.stride / self.fps,
            "episodes": rows,
            "warning": "Heuristics for manual review; static scenes are not proof of frozen video, and low detail is not proof of blur. No frames removed.",
        }
