"""Serializable analysis decisions and explicit mutations, independent of storage."""
from dataclasses import asdict, dataclass, field
from .trajectory import _json_safe


@dataclass
class EpisodeDecision:
    keep: bool = True
    reasons: list[str] = field(default_factory=list)


@dataclass
class NumericEdit:
    column: str
    start: int
    end: int
    operation: str
    parameters: dict = field(default_factory=dict)


@dataclass
class TransformPlan:
    episode_id: int
    source_frames: int
    source_identity: str
    trim: tuple[int, int] | None = None
    drop_frames: list[int] = field(default_factory=list)
    crop: dict[str, dict] = field(default_factory=dict)
    reject_episode: bool = False
    reasons: list[str] = field(default_factory=list)
    numeric: list[NumericEdit] = field(default_factory=list)
    reindex: bool = False
    preprocessing: dict[str, dict] = field(default_factory=dict)

    @property
    def changes_data(self):
        return bool(self.reject_episode or self.trim is not None or self.drop_frames
                    or self.crop or self.numeric or self.reindex)

    def to_dict(self):
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["numeric"] = [NumericEdit(**edit) for edit in value.get("numeric", [])]
        if value.get("trim") is not None:
            value["trim"] = tuple(value["trim"])
        return cls(**value)


@dataclass
class QualityReport:
    episode_id: int
    checks: dict
    groups: dict = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    episode_decision: EpisodeDecision = field(default_factory=EpisodeDecision)
    abort: bool = False

    def to_dict(self):
        return _json_safe(asdict(self))
