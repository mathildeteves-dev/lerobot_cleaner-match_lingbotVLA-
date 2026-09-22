"""Model-independent language evidence. No normalization or tokenization."""
from dataclasses import dataclass
from numbers import Integral, Real
import math


@dataclass(frozen=True)
class LanguageFeature:
    index_column: str = "task_index"
    text_columns: tuple[str, ...] = ("task", "language", "instruction", "prompt")


@dataclass(frozen=True)
class TaskInfo:
    task_index: object
    task_text: object
    task_source: str
    episode_index: int
    sample_index: int | None = None  # Original source position, survives frame filtering.


@dataclass(frozen=True)
class EpisodeTask:
    episode_index: int
    samples: tuple[TaskInfo, ...]
    evidence: tuple[TaskInfo, ...] = ()
    issues: tuple[dict, ...] = ()
    scope: str = "episode"


def valid_index(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def text_error(value):
    if value is None or (isinstance(value, Real) and math.isnan(value)):
        return "null_task_text"
    if not isinstance(value, str):
        return "wrong_type_task_text"
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return "invalid_encoding"
    if not value.strip():
        return "empty_task_text"
    return None


def report_value(value):
    """Keep invalid evidence reportable without repairing its canonical value."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    if value is None:
        return None
    if valid_index(value):
        return int(value)
    return repr(value).encode("utf-8", errors="backslashreplace").decode("utf-8")
