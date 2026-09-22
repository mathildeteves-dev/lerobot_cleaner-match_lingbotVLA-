"""Image storage cells shared by readers and writers."""
from io import BytesIO
from pathlib import Path
import numpy as np


def image_array(value, root):
    from PIL import Image
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(BytesIO(value["bytes"])) as image:
                return np.array(image)
        elif value.get("path") is not None:
            path = Path(value["path"])
            path = path if path.is_absolute() else root / path
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Image reference escapes source dataset")
            with Image.open(path) as image:
                return np.array(image)
        else:
            raise ValueError("Image storage cell has no bytes or path")
    if hasattr(value, "convert"):
        value = np.asarray(value)
    return np.asarray(value)

