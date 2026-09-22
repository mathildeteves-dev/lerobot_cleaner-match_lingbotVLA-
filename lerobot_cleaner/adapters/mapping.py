"""Model-independent mapping declarations and generic YAML grammar."""
from dataclasses import dataclass
from lerobot_cleaner.core.physical import PhysicalSemantics, PHYSICAL_FIELDS
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SourceSpec:
    key: str
    start: int | None = None
    end: int | None = None
    actual_dimension: int | None = None


@dataclass(frozen=True)
class FeatureSpec(PhysicalSemantics):
    target: str
    modality: str
    sources: tuple[SourceSpec, ...]
    dimension: int | None = None


class MappingLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Generic mapping target/source={key!r}, slice=unknown, actual dimension=unknown: duplicate or non-string YAML key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


MappingLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def mapping_error(target, source, start, end, actual, reason):
    raise ValueError(f"Mapping target={target!r}, source={source!r}, slice=[{start}:{end}], actual dimension={actual}: {reason}")


def parse_mapping(document):
    def invalid(target, reason):
        mapping_error(target, "<config>", None, None, "unknown", reason)
    if not isinstance(document, dict) or set(document) != {"canonical"}:
        invalid("<root>", "expected canonical only")
    sections = document["canonical"]
    if not isinstance(sections, dict) or not sections or set(sections) - {"state", "action"}:
        invalid("canonical", "expected state/action mappings")
    specs = []
    for modality, entries in sections.items():
        if not isinstance(entries, dict):
            invalid(modality, "expected a target mapping")
        for name, entry in entries.items():
            if not isinstance(name, str) or not name.strip():
                invalid(name, "target must be a nonempty string")
            target = name if name.startswith(modality + ".") else modality + "." + name
            dimension = None
            physical = {}
            if isinstance(entry, dict):
                entry = dict(entry)
                physical = {key: entry.pop(key) for key in PHYSICAL_FIELDS if key in entry}
                try:
                    physical = PhysicalSemantics(**physical).physical_metadata()
                except ValueError as exc:
                    invalid(target, str(exc))
            if isinstance(entry, dict) and "sources" in entry:
                if set(entry) - {"sources", "dimension"}:
                    invalid(target, "only sources and dimension are allowed")
                dimension, entry = entry.get("dimension"), entry["sources"]
            if isinstance(entry, str):
                entry = [{"source": entry}]
            elif isinstance(entry, dict):
                entry = [entry]
            if not isinstance(entry, list) or not entry:
                invalid(target, "sources must be a nonempty ordered list")
            sources = []
            for source in entry:
                if not isinstance(source, dict) or "source" not in source or set(source) - {"source", "start", "end"}:
                    invalid(target, "each source requires source and optional start/end; no step or implicit order")
                sources.append(SourceSpec(source["source"], source.get("start"), source.get("end")))
            specs.append(FeatureSpec(target, modality, tuple(sources), dimension, **physical))
    if not specs:
        invalid("canonical", "at least one feature is required")
    return tuple(specs)


def load_mapping(path):
    try:
        return parse_mapping(yaml.load(Path(path).read_text(encoding="utf-8"), Loader=MappingLoader))
    except yaml.YAMLError as exc:
        mapping_error("<root>", str(path), None, None, "unknown", str(exc))
