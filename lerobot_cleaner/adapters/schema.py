"""Canonical, storage-independent feature declarations."""
from dataclasses import dataclass

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
class FeatureSchema:
    name: str
    modality: str
    slices: tuple[FeatureSlice, ...] = ()
    camera_column: str | None = None
    subtract_state: bool = False  # Training recipe metadata; never applied implicitly.

    @property
    def width(self):
        return sum(part.end - part.start for part in self.slices)

    def extract(self, frame):
        arrays = []
        for part in self.slices:
            if part.column not in frame:
                raise ValueError(f"Missing feature column: {part.column}")
            values = vector_values(frame[part.column], part.end)
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2 or not 0 <= part.start < part.end <= values.shape[1]:
                raise ValueError(f"Invalid slice for {self.name}: {part}")
            arrays.append(values[:, part.start:part.end])
        return np.concatenate(arrays, axis=1) if arrays else np.empty((len(frame), 0))


# Compatibility for callers of the original adapter API.
Feature = FeatureSchema


@dataclass(frozen=True)
class CanonicalFeatureSchema:
    states: tuple[FeatureSchema, ...] = ()
    actions: tuple[FeatureSchema, ...] = ()
    cameras: tuple[FeatureSchema, ...] = ()
    # GR00T rules address raw indices, including unnamed vector dimensions.
    raw_vectors: bool = False
    state_column: str = "observation.state"
    action_column: str = "action"
