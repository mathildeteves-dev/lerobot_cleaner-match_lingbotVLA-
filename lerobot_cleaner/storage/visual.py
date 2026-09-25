"""Storage dispatch for canonical visuals; outputs HWC pixels in [0, 255]."""
from pathlib import Path
import numpy as np
from .images import image_array


class VisualReader:
    def __init__(self, storage, episode):
        self.storage, self.episode = storage, episode
        self._references = None

    def references(self, feature):
        """Inspect storage references without claiming pixel decoding succeeded."""
        for source in feature.sources:
            key = source.source_key
            if source.storage_dtype == "video":
                if self._references is None:
                    self._references = {r["key"]: r for r in self.storage.video_references(self.episode.episode_index)}
                ref = self._references.get(key)
                if ref is None:
                    raise ValueError(f"Missing video reference: {key}")
                path = Path(ref["path"])
                if (not path.is_file() or path.stat().st_size == 0 or
                    not np.isfinite([ref["start"], ref["end"]]).all() or ref["start"] < 0 or
                    not np.isclose(ref["end"]-ref["start"], len(self.episode.df)/self.storage.info["fps"], atol=1e-3)):
                    raise ValueError(f"Invalid video reference/interval: {key}")
            elif source.storage_dtype == "image":
                if key not in self.episode.df:
                    raise ValueError(f"Missing image column: {key}")
                root = Path(self.storage.root).resolve()
                for cell in self.episode.df[key]:
                    if cell is None:
                        raise ValueError(f"Missing image cell: {key}")
                    if isinstance(cell, dict):
                        if cell.get("bytes") is not None:
                            if not cell["bytes"]:
                                raise ValueError(f"Empty image bytes: {key}")
                        elif cell.get("path"):
                            path = Path(cell["path"])
                            path = (path if path.is_absolute() else root/path).resolve()
                            if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
                                raise ValueError(f"Invalid image reference: {key}")
                        else:
                            raise ValueError(f"Image cell has no bytes/path: {key}")
            else:
                raise ValueError(f"Unsupported visual backend: {source.storage_dtype}")

    def frames(self, feature, positions):
        sources = []
        for source in feature.sources:
            key = source.source_key
            if source.storage_dtype == "video":
                times = self.episode.df.timestamp.to_numpy()[positions]
                batch = self.storage.video_frames(self.episode.episode_index, key, times)
                batch = batch.detach().cpu().numpy() if hasattr(batch, "detach") else np.asarray(batch)
                if batch.ndim != 4 or len(batch) != len(positions):
                    raise ValueError(f"Video decoder returned invalid batch: {key}")
                values = [np.moveaxis(pixels, 0, -1) for pixels in batch]
            elif source.storage_dtype == "image":
                values = []
                for i in positions:
                    cell = self.episode.df[key].iloc[int(i)]
                    pixels = image_array(cell, Path(self.storage.root))
                    # Encoded images/PIL always decode to HWC; array cells follow metadata axes.
                    if source.layout == "CHW" and not isinstance(cell, dict) and not hasattr(cell, "convert"):
                        pixels = np.moveaxis(pixels, 0, -1)
                    if pixels.ndim == 2:
                        pixels = pixels[..., None]
                    values.append(pixels)
            else:
                raise ValueError(f"Unsupported visual backend: {source.storage_dtype}")
            normalized = []
            for pixels in values:
                if tuple(pixels.shape) != source.shape:
                    raise ValueError(f"Visual source {key!r}: resolution {pixels.shape} differs from metadata {source.shape}")
                if not np.issubdtype(pixels.dtype, np.number) or not np.isfinite(pixels).all():
                    raise ValueError(f"Nonfinite/nonnumeric image: {key}")
                if np.issubdtype(pixels.dtype, np.floating):
                    if np.any(pixels < 0) or np.any(pixels > 1):
                        raise ValueError(f"Float visual pixels must be in [0, 1]: {key}")
                    pixels = pixels * 255.
                elif pixels.dtype != np.uint8:
                    raise ValueError(f"Unsupported pixel dtype {pixels.dtype}: {key}")
                if source.start is not None:
                    pixels = pixels[:, source.start:source.end, :]
                normalized.append(pixels)
            sources.append(normalized)
        return [np.concatenate(parts, axis=1) for parts in zip(*sources)]
