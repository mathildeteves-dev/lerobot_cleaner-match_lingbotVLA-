"""Permanent pixel crop; training preprocessing is excluded from this function."""
import numpy as np


def crop_image(image, settings):
    if settings.get("purpose", "dataset") != "dataset":
        return image
    left, top, right, bottom = settings["box"]
    array = np.asarray(image)
    if array.ndim not in {2, 3} or not 0 <= left < right <= array.shape[1] or not 0 <= top < bottom <= array.shape[0]:
        raise ValueError("Planned ROI outside decoded image dimensions")
    result = array[top:bottom, left:right].copy()
    if settings.get("resize") is not None:
        from PIL import Image
        method = Image.Resampling.NEAREST if array.ndim == 2 else Image.Resampling.BILINEAR
        result = np.asarray(Image.fromarray(result).resize(tuple(settings["resize"]), resample=method))
    return result
