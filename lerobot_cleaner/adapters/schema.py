"""Canonical, storage-independent feature declarations."""
from dataclasses import dataclass, field
from lerobot_cleaner.core.physical import PhysicalSemantics
from lerobot_cleaner.core.language import LanguageFeature

import numpy as np


def vector_values(series, minimum_width):
    """Expand missing vector cells while rejecting inconsistent observed widths."""
    rows = []
    width = None
    for value in series:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0 and np.isnan(array):
            rows.append(None)
            continue
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1 or len(array) < minimum_width:
            raise ValueError("Invalid vector width")
        if width is not None and len(array) != width:
            raise ValueError("Inconsistent vector widths")
        width = len(array)
        rows.append(array)
    width = minimum_width if width is None else width
    return (np.stack([np.full(width, np.nan) if row is None else row for row in rows])
            if rows else np.empty((0, width)))


@dataclass(frozen=True)
class FeatureSlice:
    column: str
    start: int
    end: int


@dataclass(frozen=True)
class FeatureSchema(PhysicalSemantics):
    name: str
    modality: str
    # Numeric last-axis slices; for derived cameras, slices address CHW width.
    slices: tuple[FeatureSlice, ...] = ()
    camera_column: str | None = None
    subtract_state: bool = False  # Training recipe metadata; never applied implicitly.

    convert_from_state: bool = False  # LingBot marker; source is always resolved in slices.

    source_dimensions: tuple[int | None, ...] = ()

    @property
    def width(self):
        return sum(part.end - part.start for part in self.slices)

    def extract(self, frame):
        from .resolver import FeatureResolver
        return FeatureResolver.extract(self, frame)


@dataclass(frozen=True)
class VisualSource:
    source_key: str
    storage_dtype: str
    shape: tuple[int, int, int]  # Normalized HWC, independent of storage layout.
    start: int | None = None
    end: int | None = None
    layout: str = "HWC"


@dataclass(frozen=True)
class VisualFeatureSchema:
    feature_name: str
    canonical_path: str
    storage_dtype: str
    source_key: str | None
    shape: tuple[int, int, int]
    sources: tuple[VisualSource, ...]
    semantic_modality: str = field(default="visual", init=False)

    @property
    def name(self):
        return self.feature_name

    @property
    def modality(self):
        return self.semantic_modality

    @property
    def camera_column(self):
        return self.source_key

    @property
    def slices(self):
        return tuple(FeatureSlice(s.source_key, s.start, s.end)
                     for s in self.sources if s.start is not None)

    @property
    def width(self):
        return sum(part.end - part.start for part in self.slices)


VisualFeature = VisualFeatureSchema


# Compatibility for callers of the original adapter API.
Feature = FeatureSchema


@dataclass(frozen=True)
class CanonicalFeatureSchema:
    states: tuple[FeatureSchema, ...] = ()
    actions: tuple[FeatureSchema, ...] = ()
    cameras: tuple[VisualFeatureSchema, ...] = ()
    # GR00T rules address raw indices, including unnamed vector dimensions.
    raw_vectors: bool = False
    state_column: str = "observation.state"
    action_column: str = "action"
    language: LanguageFeature = field(default_factory=LanguageFeature)

    @property
    def visual(self):
        """Semantic API; cameras remains the positional/keyword compatibility field."""
        return self.cameras
