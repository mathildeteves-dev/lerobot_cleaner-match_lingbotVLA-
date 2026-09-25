"""R4: video ROI crop.

Records a per-key crop (and optional resize) on the EpisodeWork. The actual
re-encode happens in the writer, which also re-slices to the surviving frames,
guaranteeing the cropped video stays aligned with the cleaned parquet rows.
ROIs are stored as ratios so they survive resolution differences across views.
"""

from __future__ import annotations

from lerobot_cleaner.v21.rules.base import TransformRule
from lerobot_cleaner.v21.types import EpisodeWork, TransformResult, VideoTransform


class VideoRoiCropRule(TransformRule):
    name = "video_roi_crop"

    def transform(self, work: EpisodeWork, context=None) -> TransformResult:
        views = {}
        resize_wh = None
        if self.config.resize_after_crop:
            w, h = self.config.resize_after_crop
            resize_wh = (int(w), int(h))

        for config_key, roi in self.config.rois.items():
            # Map a modality video key to the dataset's original_key when needed.
            try:
                original = self.resolver.video_original_key(config_key)
            except KeyError:
                original = config_key
            if original not in work.ref.video_paths:
                continue
            transform = work.video_transforms.get(original, VideoTransform())
            crop_ratio = (roi.x_ratio, roi.y_ratio, roi.w_ratio, roi.h_ratio)
            changed = transform.crop_ratio != crop_ratio or transform.resize_wh != resize_wh
            views[original] = {"changed": changed, "crop_ratio": crop_ratio, "resize_wh": resize_wh}
            transform.crop_ratio = crop_ratio
            transform.resize_wh = resize_wh
            work.video_transforms[original] = transform
            self.stats[f"cropped:{original}"] += 1

        return TransformResult(work, any(v["changed"] for v in views.values()), {"views": views, "deferred_video_encoding": True})
