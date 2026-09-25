"""Pure visual integrity decisions over storage observations."""
from .._common import result


def check_visual(observations, decoded=False):
    failures = [row for row in observations if not row["valid"]]
    return result("visual_integrity", not failures, {
        "evaluated": True, "applicable": bool(observations), "cameras": observations, "pixel_queries": decoded,
        "scope": "sampled visual decoding and resolution" if decoded else "metadata and referenced cells/files"},
        "invalid visual references or pixels" if failures else None)
