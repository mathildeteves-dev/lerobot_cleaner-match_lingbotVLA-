"""Normalize the robot grammar in LingBot FeatureTransform (4eb34b7).

This is feature mapping, not model padding/normalization. Training joint order
belongs to data_config. convert_from_state is an inert upstream marker in this
revision: only explicit origin_keys determines the actual source.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import yaml
from .schema import FeatureSchema, FeatureSlice
from .mapping import FeatureSpec, SourceSpec
from .resolver import FeatureResolver


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    # YAML merge overrides have defined SafeLoader semantics; distinguish them
    # from two explicit spellings of the same key in a single mapping.
    seen = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
            seen.add(key)
        except TypeError as exc:
            raise ValueError("Invalid LingBot robot config: mapping keys must be scalar") from exc
        if duplicate:
            raise ValueError(f"Invalid LingBot robot config: duplicate YAML key {key!r}")
    loader.flatten_mapping(node)
    return loader.construct_mapping(node, deep=deep)


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _error(section, target, field, reason):
    raise ValueError(f"Invalid LingBot robot config: {section} target {target!r}, {field}: {reason}")


@dataclass(frozen=True)
class SourceSliceSpec:
    origin_key: str
    start: int
    end: int
    # Preserve the YAML spelling as well as resolved Python slicing semantics.
    declared_start: int | None = None
    declared_end: int | None = None
    actual_dimension: int | None = None


@dataclass(frozen=True)
class NumericFeatureSpec:
    target_name: str
    section: str
    sources: tuple[SourceSliceSpec, ...]
    direct: bool = False
    subtract_state: bool = False
    convert_from_state: bool = False

    @property
    def dimension(self):
        return sum(s.end - s.start for s in self.sources)

    @property
    def subtract_state_source(self):
        return self.target_name.replace("action.", "observation.state.") if self.subtract_state else None

    def normalized(self):
        return FeatureSpec(self.target_name, "state" if self.section == "states" else "action",
            tuple(SourceSpec(s.origin_key, s.start, s.end, s.actual_dimension) for s in self.sources))

    def canonical(self):
        return replace(FeatureResolver().canonical(self.normalized()),
            subtract_state=self.subtract_state, convert_from_state=self.convert_from_state)


@dataclass(frozen=True)
class ImageFeatureSpec:
    target_name: str
    origin_key: str | None
    direct: bool = False
    sources: tuple[SourceSliceSpec, ...] = ()

    @property
    def dimension(self):
        return sum(s.end - s.start for s in self.sources) if self.sources else None

    def canonical(self):
        return FeatureSchema(self.target_name, "visual",
            tuple(FeatureSlice(s.origin_key, s.start, s.end) for s in self.sources),
            camera_column=self.origin_key)


@dataclass(frozen=True)
class LingBotRobotConfigSchema:
    states: tuple[NumericFeatureSpec, ...]
    actions: tuple[NumericFeatureSpec, ...]
    images: tuple[ImageFeatureSpec, ...]
    upstream_revision: str = "4eb34b7693a0565c67433f8fac9c59a2e67eb60b"

    @classmethod
    def from_yaml(cls, path, features, *, allow_missing_images=False):
        try:
            with Path(path).open(encoding="utf-8") as handle:
                robot = yaml.load(handle, Loader=_UniqueLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid LingBot robot config: {path}: {exc}") from exc
        return cls.parse(robot, features, allow_missing_images=allow_missing_images)

    @classmethod
    def parse(cls, robot, features, *, allow_missing_images=False):
        if not isinstance(robot, dict) or set(robot) - {"states", "actions", "images"}:
            _error("root", "<root>", "sections", "expected states/actions/images lists only")
        parsed = {}
        all_targets = set()
        mutable_mappings = set()
        for section in ("states", "actions", "images"):
            entries = robot.get(section, [])
            if not isinstance(entries, list):
                _error(section, "<section>", "entries", "expected a list")
            output = []
            for entry in entries:
                direct = isinstance(entry, str)
                if direct:
                    name, mapping = entry, {}
                    if section == "actions":
                        _error(section, name, "entry", "official actions require a mapping, not a direct key")
                elif isinstance(entry, dict) and len(entry) == 1:
                    name, mapping = next(iter(entry.items()))
                else:
                    _error(section, "<entry>", "entry", "expected a direct key or single-target mapping")
                if not isinstance(name, str) or not name or name in all_targets:
                    _error(section, name, "target", "expected a unique nonempty target name")
                all_targets.add(name)
                if not isinstance(mapping, dict):
                    _error(section, name, "mapping", "expected a mapping")
                allowed = {"origin_keys"}
                if section == "actions":
                    allowed |= {"subtract_state", "convert_from_state"}
                if set(mapping) - allowed:
                    _error(section, name, "fields", f"unsupported fields {sorted(set(mapping)-allowed)}; use origin_keys (plural)")
                for flag in ("subtract_state", "convert_from_state"):
                    if flag in mapping and type(mapping[flag]) is not bool:
                        _error(section, name, flag, "expected a boolean")
                if section == "actions" or isinstance(mapping.get("origin_keys"), list):
                    if id(mapping) in mutable_mappings:
                        _error(section, name, "mapping", "shared mapping aliases are mutated by upstream; use independent mappings")
                    mutable_mappings.add(id(mapping))
                convert = mapping.get("convert_from_state", False)
                if not direct and "origin_keys" not in mapping:
                    reason = ("upstream records this flag but does not generate an action from state; explicit origin_keys is required"
                              if convert else "no source is produced by the official mapping implementation")
                    _error(section, name, "convert_from_state/origin_keys" if convert else "origin_keys", reason)
                origins = name if direct else mapping["origin_keys"]
                if section == "images" and isinstance(origins, str):
                    if not allow_missing_images and features.get(origins, {}).get("dtype") not in {"image", "video"}:
                        _error(section, name, "origin_keys", f"missing image/video source {origins!r}")
                    output.append(ImageFeatureSpec(name, origins, direct))
                    continue
                sources = []
                if isinstance(origins, str):
                    parts = [(origins, None)]
                elif isinstance(origins, list) and origins:
                    parts = []
                    for item in origins:
                        if not isinstance(item, dict) or not item:
                            _error(section, name, "origin_keys", "list entries must map source keys to start/end slices (not strings)")
                        parts.extend(item.items())  # Official inner mapping iteration order matters too.
                else:
                    _error(section, name, "origin_keys", "expected a string or nonempty list of slice mappings")
                for key, bounds in parts:
                    if not isinstance(key, str):
                        _error(section, name, "origin_keys", "source keys must be strings")
                    # Official list grammar uses '*' suffixes for repeated source keys.
                    column = key.split("*")[0] if bounds is not None else key
                    spec = features.get(column, {})
                    shape = spec.get("shape", [])
                    dtype = spec.get("dtype", "")
                    if section == "images":
                        try:
                            visual = FeatureResolver().resolve_visual(FeatureSchema(name, "visual", camera_column=column), features)
                            shape = visual.shape
                            width = shape[1]  # Official tensors are CHW: last-axis slices address width.
                            if sources:
                                first = FeatureResolver().resolve_visual(FeatureSchema(name, "visual", camera_column=sources[0].origin_key), features).shape
                                if (shape[0], shape[2]) != (first[0], first[2]):
                                    raise ValueError("image concatenation requires equal height/channels")
                        except ValueError as exc:
                            _error(section, name, "origin_keys", str(exc))
                    else:
                        try:
                            resolved = FeatureResolver().resolve_source(name, SourceSpec(column), features)
                        except ValueError as exc:
                            _error(section, name, "origin_keys", str(exc))
                        width = resolved.actual_dimension
                    if bounds is None:
                        if not isinstance(origins, str):
                            _error(section, name, "slice", "expected start/end mapping")
                        start, end = 0, width
                        declared_start, declared_end = None, None
                    else:
                        if not isinstance(bounds, dict) or set(bounds) != {"start", "end"}:
                            _error(section, name, "slice", "expected start and end only; no step syntax")
                        declared_start, declared_end = bounds["start"], bounds["end"]
                        if any(type(v) is not int for v in (declared_start, declared_end)):
                            _error(section, name, "slice", "start/end must be integers; official reverse mapping rejects null endpoints")
                        if section == "images":
                            start, end, _ = slice(declared_start, declared_end).indices(width)
                    if section == "images" and start >= end:
                        _error(section, name, "slice", f"empty slice of {column!r}; cleaner requires positive feature dimensions")
                    if section != "images":
                        # Grammar-specific '*' aliases and Python bounds are normalized
                        # here; source validation and dimension inference are shared.
                        try:
                            resolved = FeatureResolver().resolve_source(name,
                                SourceSpec(column, declared_start, declared_end), features, slice_mode="python")
                        except ValueError as exc:
                            _error(section, name, "slice", str(exc))
                        start, end = resolved.start, resolved.end
                    sources.append(SourceSliceSpec(column, start, end, declared_start, declared_end, width))
                if section == "images":
                    output.append(ImageFeatureSpec(name, None, False, tuple(sources)))
                    continue
                subtract = mapping.get("subtract_state", False)
                if subtract and ("end.position" in name or "effector.position" in name):
                    _error(section, name, "subtract_state", "forbidden for end.position/effector.position by LingBot")
                output.append(NumericFeatureSpec(name, section, tuple(sources), direct, subtract, convert))
            parsed[section] = tuple(output)
        schema = cls(**parsed)
        states = {s.target_name: s for s in schema.states}
        for action in schema.actions:
            if action.subtract_state:
                state = states.get(action.subtract_state_source)
                if state is None or state.dimension != action.dimension:
                    _error("actions", action.target_name, "subtract_state",
                           f"requires state {action.subtract_state_source!r} with dimension {action.dimension}")
        return schema

    def convert_features(self, item, *, subtract_state=False):
        """Explicit pre-normalization training view; never mutate raw input.

        Canonical cleaner extraction stays absolute. Call this separately to
        reproduce FeatureTransform.apply(do_nomalize=False), including deltas.
        Padding, image processing and training joint order are outside this API.
        """
        output = {}
        for feature in self.states + self.actions:
            output[feature.target_name] = FeatureResolver.extract_item(feature.normalized(), item)
        for feature in self.images:
            output[feature.target_name] = (np.concatenate([
                np.asarray(item[s.origin_key])[..., s.start:s.end] for s in feature.sources], axis=-1)
                if feature.sources else np.array(item[feature.origin_key], copy=True))
        if subtract_state:
            for action in self.actions:
                if action.subtract_state:
                    output[action.target_name] = output[action.target_name] - output[action.subtract_state_source]
        return output
