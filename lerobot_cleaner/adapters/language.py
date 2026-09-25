"""Resolve task tables and sample/episode fields without changing source text."""
from collections import defaultdict

from lerobot_cleaner.core.language import EpisodeTask, LanguageFeature, TaskInfo, valid_index


def task_catalog(table):
    """Official v3 indexed table or legacy v2 task records; None means no table."""
    if table is None:
        return None
    if hasattr(table, "iterrows"):
        return tuple((row.get("task_index"), row.get("task", label))
                     for label, row in table.iterrows())
    return tuple((row.get("task_index"), row.get("task")) for row in table)


def resolve_language(frame, episode_id, metadata, catalog=None, feature=None, positions=None):
    feature = feature or LanguageFeature()
    positions = list(range(len(frame))) if positions is None else positions
    mapping = defaultdict(list)
    issues, evidence, samples = [], [], []
    if catalog is not None:
        for index, text in catalog:
            if valid_index(index):
                mapping[int(index)].append(text)
            else:
                issues.append({"code": "invalid_table_index", "task_index": index})
        for index, entries in mapping.items():
            if len(entries) != 1:
                issues.append({"code": "duplicate_table_index", "task_index": index})
        expected = metadata.get("task_table_expected_count")
        if expected is not None and (len(catalog) != expected or set(mapping) != set(range(expected))):
            issues.append({"code": "inconsistent_task_table", "expected_count": expected})
    declared = []
    for key in (*feature.text_columns, "tasks"):
        if key not in metadata:
            continue
        values = metadata[key]
        if key == "tasks" and hasattr(values, "tolist"):
            values = values.tolist()
        values = values if key == "tasks" and isinstance(values, (list, tuple)) else [values]
        for value in values:
            item = TaskInfo(metadata.get(feature.index_column), value, "episode metadata:" + key, episode_id)
            declared.append(item)
            evidence.append(item)
    dataset_declared = [TaskInfo(None, value, "dataset metadata:" + key, episode_id)
                        for key, value in metadata.get("dataset_language", {}).items()]
    evidence.extend(dataset_declared)
    if not declared:
        declared.extend(dataset_declared)
    columns = [key for key in feature.text_columns if key in frame]
    has_index = feature.index_column in frame
    indexed = catalog is not None or has_index or feature.index_column in metadata
    # Column iteration preserves integer task IDs (iterrows may cast them to float).
    indexes = frame[feature.index_column].tolist() if has_index else [metadata.get(feature.index_column)] * len(frame)
    text_columns = {key: frame[key].tolist() for key in columns}
    for position, index in zip(positions, indexes):
        local = len(samples)
        candidates = [TaskInfo(index, values[local], "sample field:" + key, episode_id, position)
                      for key, values in text_columns.items()]
        evidence.extend(candidates)
        table_item = None
        if indexed:
            code = None
            if index is None:
                code = "missing_task_index"
            elif not valid_index(index):
                code = "invalid_task_index_type"
            elif metadata.get("task_table_expected_count") is not None and index >= metadata["task_table_expected_count"]:
                code = "task_index_out_of_range"
            elif catalog is None or int(index) not in mapping:
                code = "unresolved_task_index"
            elif len(mapping[int(index)]) != 1:
                code = "ambiguous_task_index"
            else:
                table_item = TaskInfo(index, mapping[int(index)][0], "task table", episode_id, position)
                evidence.append(table_item)
            if code:
                issues.append({"code": code, "sample_index": position, "task_index": index})
        # A present but invalid authoritative value is evidence, never silently repaired.
        selected = table_item or (candidates[0] if candidates else None)
        if selected is None and declared:
            # Equivalent declarations in task/language are one instruction. Distinct
            # declarations without a sample mapping are ambiguous, not a fallback.
            distinct = {repr(item.task_text) for item in declared}
            if len(distinct) == 1:
                item = declared[0]
                selected = TaskInfo(index, item.task_text, item.task_source, episode_id, position)
            else:
                issues.append({"code": "ambiguous_episode_task", "sample_index": position})
        samples.append(selected or TaskInfo(index, None, "unresolved", episode_id, position))
        comparable = [item.task_text for item in candidates + ([table_item] if table_item else [])
                      if isinstance(item.task_text, str)]
        if len(set(comparable)) > 1:
            issues.append({"code": "inconsistent_mapping", "sample_index": position})
        if selected is not None and declared and isinstance(selected.task_text, str):
            # `tasks` declares the set of instructions used in the episode. Singular
            # fields declare one instruction and must agree independently.
            groups = defaultdict(list)
            for item in declared:
                groups[item.task_source].append(item.task_text)
            if any(selected.task_text not in [v for v in values if isinstance(v, str)]
                   for values in groups.values()):
                issues.append({"code": "inconsistent_episode_mapping", "sample_index": position})
        if feature.index_column in metadata and valid_index(index):
            expected = metadata[feature.index_column]
            if not valid_index(expected) or int(expected) != int(index):
                issues.append({"code": "inconsistent_episode_index", "sample_index": position})
    return EpisodeTask(episode_id, tuple(samples), tuple(evidence), tuple(issues),
                       "sample" if columns or has_index else "episode")
