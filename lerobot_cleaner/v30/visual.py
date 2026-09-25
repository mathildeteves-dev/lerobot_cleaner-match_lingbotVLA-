"""Canonical visual quality orchestration, independent of pixel storage format."""
from dataclasses import asdict
from lerobot_cleaner.core.quality.vision.pixels import pixel_metrics, check_black, check_brightness
import numpy as np
from lerobot_cleaner.storage.visual import VisualReader
from lerobot_cleaner.core.quality.integrity.visual import check_visual
from lerobot_cleaner.core.quality.vision.blur import blur_variance, check_blur


def visual_findings(storage, episode, options, features):
    reader = VisualReader(storage, episode)
    observations, variances, samples = [], [], []
    for feature in features:
        item = {**asdict(feature), "key": feature.source_key, "valid": True, "sampled_frames": 0}
        try:
            reader.references(feature)
            if options.decode:
                positions = np.arange(0, len(episode.df), options.sample_stride)
                for start in range(0, len(positions), options.batch_frames):
                    ids = positions[start:start+options.batch_frames]
                    for pixels in reader.frames(feature, ids):
                        variances.append(blur_variance(pixels))
                        samples.append(pixel_metrics(pixels, options.black_level))
                        item["sampled_frames"] += 1
                if item["sampled_frames"] != len(positions) or not len(positions):
                    raise ValueError("No pixels or incomplete visual samples")
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError) as exc:
            item.update(valid=False, error=str(exc))
        observations.append(item)
    integrity = check_visual(observations, options.decode).to_dict()
    # Read-only report alias; policy iteration only evaluates the canonical rule once.
    findings = {"visual_integrity": integrity,
                "video_integrity": {**integrity, "rule": "video_integrity", "alias_of": "visual_integrity"}}
    if options.blur_variance is not None:
        finding = check_blur(variances, options.blur_variance, options.max_blur_ratio)
        if any(not row["valid"] for row in observations):
            finding.passed = False
            finding.metrics["evaluated"] = False
            finding.message = "Incomplete visual decoding; blur result is unavailable"
        findings["blur"] = finding.to_dict()
    complete = all(row["valid"] for row in observations)
    if options.black_level is not None:
        findings["black_frame"] = check_black(samples, options.max_black_ratio, complete).to_dict()
    if options.min_brightness is not None or options.max_brightness is not None:
        findings["brightness"] = check_brightness(samples, options.min_brightness, options.max_brightness, complete).to_dict()
    return findings
