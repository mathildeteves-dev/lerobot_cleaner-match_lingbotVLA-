"""Video findings from storage observations; this checker never opens files."""
from .._common import result


def check_video(observations, decoded=False):
    failures = [row for row in observations if not row["valid"]]
    return result("video_integrity", not failures, {"evaluated": True,
        "cameras": observations, "pixel_queries": decoded,
        "scope": "sampled official decoding" if decoded else "metadata and referenced files"},
        None if not failures else "invalid video references or decoding")
