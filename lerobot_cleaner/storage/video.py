"""Physical video inspection belongs to storage, not feature semantics.

Normal timestamp queries use OfficialStorage.video_frames and the official
LeRobot decoder. This full-file iterator additionally exposes original PTS for
integrity evaluation; timestamp-selected tensors cannot prove PTS continuity.
"""
from contextlib import contextmanager


@contextmanager
def decoded_video(path):
    try:
        import av
    except ImportError as exc:
        raise ImportError("Full video integrity inspection requires '.[v3-video]'") from exc
    with av.open(str(path)) as container:
        yield container.decode(container.streams.video[0])
