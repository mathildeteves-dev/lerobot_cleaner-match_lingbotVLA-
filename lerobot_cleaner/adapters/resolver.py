"""Resolve source adapters into a validated canonical feature schema."""
from .schema import CanonicalFeatureSchema, FeatureSchema, FeatureSlice, VisualFeatureSchema, VisualSource, vector_values
from lerobot_cleaner.core.physical import PhysicalSemantics
from .mapping import FeatureSpec, SourceSpec, mapping_error
import numpy as np
from lerobot_cleaner.core.language import LanguageFeature


class FeatureResolver:
    def resolve(self, adapter):
        groups = (tuple(adapter.get_state_features()), tuple(adapter.get_action_features()),
                  tuple(self.resolve_visual(f, adapter.info["features"])
                        for f in adapter.get_camera_features()))
        for features, modality in zip(groups, ("state", "action", "visual")):
            names = set()
            for feature in features:
                if feature.name in names or feature.modality != modality:
                    raise ValueError(f"Duplicate or mismatched feature: {feature.name}")
                names.add(feature.name)
                if modality == "visual":
                    if not feature.camera_column and not feature.slices:
                        raise ValueError(f"Missing camera column: {feature.name}")
                elif not feature.slices:
                    raise ValueError(f"Missing slices: {feature.name}")
                for part in feature.slices:
                    if not part.column or not 0 <= part.start < part.end:
                        raise ValueError(f"Invalid feature slice: {part}")
        paths = [feature.canonical_path for feature in groups[2]]
        if len(paths) != len(set(paths)):
            raise ValueError("Conflicting canonical visual paths: " + repr(paths))
        quality = getattr(getattr(adapter, "config", None), "quality", None)
        schema = CanonicalFeatureSchema(
            *groups, raw_vectors=getattr(adapter, "raw_vectors", False),
            state_column=getattr(quality, "state_column", "observation.state"),
            action_column=getattr(quality, "action_column", "action"),
            language=getattr(adapter, "get_language_feature", LanguageFeature)())
        return schema

    def resolve_visual(self, feature, metadata):
        """Resolve a direct visual source or ordered width slices, without model semantics."""
        name = feature.name
        direct = feature.camera_column
        parts = (FeatureSlice(direct, None, None),) if direct else feature.slices
        if not parts:
            raise ValueError(f"Visual feature {name!r}: missing source")
        sources = []
        for part in parts:
            spec = metadata.get(part.column, {})
            dtype, shape = spec.get("dtype"), spec.get("shape")
            prefix = f"Visual feature {name!r}, source {part.column!r}, slice [{part.start}:{part.end}], shape {shape!r}: "
            if dtype not in {"video", "image"}:
                raise ValueError(prefix + "missing or unsupported visual storage dtype")
            if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(type(d) is not int or d <= 0 for d in shape):
                raise ValueError(prefix + "expected positive three-dimensional image shape")
            names = spec.get("names")
            layout = "HWC"
            if names is not None:
                if list(names) in (["channels", "height", "width"], ["channel", "height", "width"]):
                    layout = "CHW"
                elif list(names) not in (["height", "width", "channels"], ["height", "width", "channel"]):
                    raise ValueError(prefix + "unrecognized visual axis names")
            normalized = tuple(shape[1:]) + (shape[0],) if layout == "CHW" else tuple(shape)
            if not direct and (type(part.start) is not int or type(part.end) is not int or not 0 <= part.start < part.end <= normalized[1]):
                raise ValueError(prefix + "empty or out-of-bounds width slice")
            sources.append(VisualSource(part.column, dtype, normalized, part.start, part.end, layout))
        if any((s.shape[0], s.shape[2]) != (sources[0].shape[0], sources[0].shape[2]) for s in sources):
            raise ValueError(f"Visual feature {name!r}: concatenation requires matching height/channels")
        shape = sources[0].shape if direct else (sources[0].shape[0], sum(s.end-s.start for s in sources), sources[0].shape[2])
        dtype = sources[0].storage_dtype if all(s.storage_dtype == sources[0].storage_dtype for s in sources) else "mixed"
        suffix = name
        for prefix in ("observation.images.", "visual.", "video."):
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix):]
                break
        return VisualFeatureSchema(name, "visual." + suffix, dtype, direct, shape, tuple(sources))

    def resolve_source(self, target, source, features, *, slice_mode="strict"):
        """Resolve a numeric source. Grammar adapters choose strict or Python bounds."""
        if slice_mode not in {"strict", "python"}:
            raise ValueError("Unknown slice mode")
        key, start, end = source.key, source.start, source.end
        metadata = features.get(key, {}) if isinstance(key, str) else {}
        shape, dtype = metadata.get("shape", []), metadata.get("dtype", "")
        actual = shape[0] if isinstance(shape, (list, tuple)) and len(shape) == 1 else shape
        def fail(reason):
            mapping_error(target, key, start, end, actual, reason)
        if not isinstance(key, str) or not key or key not in features:
            fail("source does not exist")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 1 or
                type(shape[0]) is not int or shape[0] < 1 or
                not isinstance(dtype, str) or not dtype.startswith(("float", "int", "uint", "bool"))):
            fail("source must declare a positive one-dimensional numeric vector")
        if any(value is not None and type(value) is not int for value in (start, end)):
            fail("slice endpoints must be integers")
        if slice_mode == "python":
            first, stop, _ = slice(start, end).indices(actual)
        else:
            first, stop = (0 if start is None else start), (actual if end is None else end)
            if first < 0 or stop < 0 or first > actual or stop > actual:
                fail("slice out of bounds")
        if first >= stop:
            fail("empty or reversed slice")
        return SourceSpec(key, first, stop, actual)

    def resolve_mapping(self, specs, features, *, slice_mode="strict"):
        output, targets = [], set()
        for spec in specs:
            if not isinstance(spec.target, str) or not spec.target.strip() or spec.target in targets:
                mapping_error(spec.target, "<sources>", None, None, "unknown", "empty or conflicting target")
            targets.add(spec.target)
            if spec.modality not in {"state", "action"}:
                mapping_error(spec.target, "<sources>", None, None, "unknown", "unsupported numeric modality")
            if not isinstance(spec.sources, (list, tuple)) or not spec.sources:
                mapping_error(spec.target, "<sources>", None, None, "unknown", "concat requires a nonempty ordered sequence")
            sources = tuple(self.resolve_source(spec.target, source, features, slice_mode=slice_mode) for source in spec.sources)
            output.append(self.canonical(FeatureSpec(spec.target, spec.modality, sources, spec.dimension, **spec.physical_metadata())))
        return tuple(output)

    def canonical(self, spec):
        """Materialize a normalized numeric mapping, preserving source order."""
        width = sum(source.end - source.start for source in spec.sources)
        if spec.dimension is not None and (type(spec.dimension) is not int or spec.dimension != width):
            mapping_error(spec.target, [s.key for s in spec.sources],
                [s.start for s in spec.sources], [s.end for s in spec.sources], width,
                f"concatenated dimension differs from expected {spec.dimension!r}")
        return FeatureSchema(spec.target, spec.modality,
            tuple(FeatureSlice(s.key, s.start, s.end) for s in spec.sources),
            source_dimensions=tuple(s.actual_dimension for s in spec.sources), **spec.physical_metadata())

    @staticmethod
    def extract(feature, frame):
        arrays = []
        for i, part in enumerate(feature.slices):
            expected = feature.source_dimensions[i] if feature.source_dimensions else None
            if part.column not in frame:
                mapping_error(feature.name, part.column, part.start, part.end, "missing", "source column is absent")
            try:
                values = vector_values(frame[part.column], expected or part.end)
            except (ValueError, TypeError) as exc:
                dimensions = sorted({str(np.asarray(value).shape) for value in frame[part.column]})
                mapping_error(feature.name, part.column, part.start, part.end, dimensions, str(exc))
            if expected is not None and values.shape[1] != expected:
                mapping_error(feature.name, part.column, part.start, part.end, values.shape[1], f"source dimension differs from metadata {expected}")
            if not 0 <= part.start < part.end <= values.shape[1]:
                mapping_error(feature.name, part.column, part.start, part.end, values.shape[1], "invalid runtime slice")
            arrays.append(values[:, part.start:part.end])
        return np.concatenate(arrays, axis=1) if arrays else np.empty((len(frame), 0))

    @staticmethod
    def extract_item(spec, item):
        """Apply normalized mappings to vectors or batched vectors on the last axis."""
        arrays, leading = [], None
        for source in spec.sources:
            if source.key not in item:
                mapping_error(spec.target, source.key, source.start, source.end, "missing", "source is absent")
            values = np.asarray(item[source.key])
            actual = values.shape[-1] if values.ndim else 0
            if (values.ndim < 1 or not 0 <= source.start < source.end <= actual or
                    source.actual_dimension is not None and actual != source.actual_dimension):
                mapping_error(spec.target, source.key, source.start, source.end, actual, "source dimension or slice mismatch")
            if leading is not None and values.shape[:-1] != leading:
                mapping_error(spec.target, source.key, source.start, source.end, values.shape, "concat leading dimensions differ")
            leading = values.shape[:-1]
            arrays.append(values[..., source.start:source.end])
        return np.concatenate(arrays, axis=-1)


    @staticmethod
    def default_schema(state_width, action_width, state_column="observation.state", action_column="action"):
        """Whole-vector declarations for the legacy frame-only bridge; no inferred parts."""
        def feature(name, modality, width):
            return (FeatureSchema(name, modality, (FeatureSlice(name, 0, width),)),) if width else ()
        return CanonicalFeatureSchema(states=feature(state_column, "state", state_width),
            actions=feature(action_column, "action", action_width),
            state_column=state_column, action_column=action_column)

    def resolve_selection(self, schema, requested, *, widths=None):
        """Select named numeric features in schema order, not request/hash order.

        Packed schemas use cumulative widths. Raw-vector schemas preserve original
        slice positions (including unnamed gaps), matching EpisodeBuilder.arrays.
        """
        if not isinstance(requested, (tuple, list)) or not requested:
            raise ValueError("canonical feature selection must be a nonempty ordered list")
        for i, name in enumerate(requested):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"canonical feature {name!r} must be a nonempty name")
            if name in requested[:i]:
                raise ValueError(f"canonical feature {name!r} is repeated")
        numeric = [(source, feature) for source, features in (("state", schema.states), ("action", schema.actions))
                   for feature in features]
        chosen_source = None
        for name in requested:
            matches = [(source, feature) for source, feature in numeric if feature.name == name]
            if any(feature.name == name for feature in schema.cameras):
                raise ValueError(f"canonical feature {name!r} belongs to visual, not an allowed numeric modality")
            if not matches:
                if name in {"task", "language", "instruction", "prompt", "task_index"}:
                    raise ValueError(f"canonical feature {name!r} is not a state/action feature")
                raise ValueError(f"canonical feature {name!r} does not exist")
            if len(matches) != 1:
                raise ValueError(f"canonical feature {name!r} is ambiguous in the schema")
            source, feature = matches[0]
            if feature.modality != source:
                raise ValueError(f"canonical feature {name!r} has mismatched modality {feature.modality!r}")
            if chosen_source is not None and chosen_source != source:
                raise ValueError(f"canonical feature {name!r} cannot combine {source} with {chosen_source}")
            chosen_source = source
        features = schema.states if chosen_source == "state" else schema.actions
        raw_column = schema.state_column if chosen_source == "state" else schema.action_column
        offset, columns, selected = 0, [], []
        for feature in features:
            if any(type(part.start) is not int or type(part.end) is not int or not 0 <= part.start < part.end for part in feature.slices):
                raise ValueError(f"canonical feature {feature.name!r} has an invalid source slice")
            width = feature.width
            if type(width) is not int or width <= 0:
                raise ValueError(f"canonical feature {feature.name!r} has invalid dimension {width!r}")
            if feature.name in requested:
                if schema.raw_vectors:
                    selected_columns = []
                    for part in feature.slices:
                        if part.column != raw_column or not 0 <= part.start < part.end:
                            raise ValueError(f"canonical feature {feature.name!r} cannot address raw {chosen_source} vector {raw_column!r}")
                        selected_columns.extend(range(part.start, part.end))
                else:
                    selected_columns = list(range(offset, offset + width))
                for index in selected_columns:
                    if index in columns:
                        raise ValueError(f"canonical feature {feature.name!r} overlaps an already selected column {index}")
                    columns.append(index)
                selected.append({"feature": feature.name, "canonical_path": feature.name,
                    "source_key": feature.slices[0].column if len({p.column for p in feature.slices}) == 1 else None,
                    **feature.physical_metadata(), "columns": selected_columns, "dimension": width,
                    "sources": [{"column": part.column, "start": part.start, "end": part.end} for part in feature.slices]})
            offset += width
        if not columns:
            raise ValueError(f"canonical feature {requested!r} resolves to empty columns")
        if widths is not None:
            actual = widths[chosen_source]
            if max(columns) >= actual or (not schema.raw_vectors and offset != actual):
                raise ValueError(f"canonical feature {requested!r} layout is inconsistent with {chosen_source} width {actual}")
        return {"source": chosen_source, "columns": columns, "dimension": len(columns),
                "feature_order": [item["feature"] for item in selected], "feature_ranges": selected,
                "ordering": "canonical_schema", "layout": "raw_vector_slices" if schema.raw_vectors else "packed_features"}

    def column_semantics(self, schema, source, width):
        """Bind physical metadata to the same packed/raw layout as numeric resolution."""
        features = schema.states if source == "state" else schema.actions
        result = [PhysicalSemantics() for _ in range(width)]
        assigned = {}
        for feature in features:
            selection = self.resolve_selection(schema, [feature.name], widths={source: width})
            for column in selection["columns"]:
                metadata = feature.physical()
                if column in assigned and assigned[column] != metadata:
                    raise ValueError(f"Conflicting physical metadata for {source} column {column}: {feature.name}")
                assigned[column] = metadata
                result[column] = metadata
        return tuple(result)
