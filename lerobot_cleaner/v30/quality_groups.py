"""Quality selections bind to canonical layout; no slicing grammar lives here."""
from lerobot_cleaner.adapters.resolver import FeatureResolver


class QualityGroupResolver:
    def resolve(self, group, schema, widths):
        requested = [group.feature] if group.feature is not None else group.features
        try:
            if requested is not None:
                if schema is None:
                    raise ValueError(f"canonical feature {requested!r} requires a CanonicalFeatureSchema")
                resolved = FeatureResolver().resolve_selection(schema, requested, widths=widths)
            else:
                columns = list(group.columns)
                width = widths[group.source]
                if not columns or any(index < 0 or index >= width for index in columns):
                    raise ValueError(f"legacy columns {columns} exceed {group.source} width {width}")
                resolved = {"source": group.source, "columns": columns, "dimension": len(columns),
                            "feature_order": [], "feature_ranges": [], "ordering": "configured_columns",
                            "layout": "legacy_columns"}
            if schema is not None:
                physical = FeatureResolver().column_semantics(schema, resolved["source"], widths[resolved["source"]])
                resolved["physical_semantics"] = [physical[i].physical_metadata() for i in resolved["columns"]]
        except ValueError as exc:
            raise ValueError(f"Quality group {group.name!r}: {exc}.") from exc
        return {"group": group.name, "requested_features": list(requested or []),
                "selection_mode": "semantic" if requested is not None else "legacy", "resolved": resolved}
