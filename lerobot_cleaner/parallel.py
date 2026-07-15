"""Per-episode worker: run rules, stage cleaned parquet + re-encoded videos.

Heavy work (rule application + ffmpeg re-encode) runs in worker processes and
writes artifacts to a staging directory keyed by the *source* episode index.
The main process then assigns final indices, finalizes parquet index columns,
moves videos into place, and accumulates exact streaming stats sequentially.

Returning staged file paths (not DataFrames) keeps inter-process payloads tiny.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot_cleaner.dataset.reader import EpisodeRef
from lerobot_cleaner.rules.base import Rule
from lerobot_cleaner.types import EpisodeWork
from lerobot_cleaner.video_utils import copy_or_reencode


@dataclass
class EpisodeResult:
    src_index: int
    dropped: bool
    drop_reason: Optional[str]
    length: int
    tasks: list[str]
    staged_parquet: Optional[str]
    staged_videos: dict[str, str]  # original_key -> staged path
    notes: dict[str, str]
    rule_stats: dict[str, Counter] = field(default_factory=dict)


def process_episode(
    ref: EpisodeRef,
    rules: list[Rule],
    staging_dir: Path,
    codec: str,
    fps: float,
    video_keys: list[str],
) -> EpisodeResult:
    """Run all rules on one episode and stage its cleaned artifacts."""
    df = ref.load_parquet()
    work = EpisodeWork(ref=ref, df=df, keep_indices=list(range(len(df))))

    per_rule_stats: dict[str, Counter] = {}
    for rule in rules:
        rule.stats = Counter()
        rule.apply(work)
        per_rule_stats[rule.name] = Counter(rule.stats)
        if work.dropped:
            break

    if work.dropped:
        return EpisodeResult(
            src_index=ref.episode_index,
            dropped=True,
            drop_reason=work.drop_reason,
            length=0,
            tasks=ref.tasks,
            staged_parquet=None,
            staged_videos={},
            notes=work.notes,
            rule_stats=per_rule_stats,
        )

    ep_stage = staging_dir / f"ep_{ref.episode_index:06d}"
    ep_stage.mkdir(parents=True, exist_ok=True)

    staged_parquet = ep_stage / "data.parquet"
    work.df.to_parquet(staged_parquet, index=False)

    staged_videos: dict[str, str] = {}
    for vkey, src_path in ref.video_paths.items():
        transform = work.video_transforms.get(vkey)
        crop_ratio = transform.crop_ratio if transform else None
        resize_wh = transform.resize_wh if transform else None
        dst = ep_stage / f"{vkey}.mp4"
        copy_or_reencode(
            src_path,
            dst,
            keep_indices=work.keep_indices,
            crop_ratio=crop_ratio,
            resize_wh=resize_wh,
            codec=codec,
            fps=fps,
        )
        staged_videos[vkey] = str(dst)

    return EpisodeResult(
        src_index=ref.episode_index,
        dropped=False,
        drop_reason=None,
        length=len(work.df),
        tasks=ref.tasks,
        staged_parquet=str(staged_parquet),
        staged_videos=staged_videos,
        notes=work.notes,
        rule_stats=per_rule_stats,
    )
