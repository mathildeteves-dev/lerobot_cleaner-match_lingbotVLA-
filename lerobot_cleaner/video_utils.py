"""Thin ffmpeg/ffprobe wrappers for video integrity, frame extraction, re-encode.

We shell out to ffmpeg/ffprobe rather than depend on a python binding, so the
only runtime requirement is a working ffmpeg in PATH. All functions raise
:class:`VideoError` on failure so the pipeline can decide drop-vs-warn.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class VideoError(RuntimeError):
    pass


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise VideoError("ffmpeg/ffprobe not found in PATH. Install ffmpeg.")


def ffprobe_streams(path: str | Path) -> dict:
    """Return parsed ffprobe JSON for the first video stream + format."""
    _require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def count_frames(path: str | Path) -> int:
    """Count decodable frames. Falls back to nb_frames if exact count is absent."""
    _require_ffmpeg()
    # Exact count by demuxing packets of the video stream.
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-print_format", "csv=p=0", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip().isdigit():
        return int(proc.stdout.strip())
    raise VideoError(f"Could not count frames for {path}: {proc.stderr.strip()}")


def is_decodable(path: str | Path) -> bool:
    """Full decode pass; returns False if any frame fails to decode."""
    _require_ffmpeg()
    if not Path(path).exists():
        return False
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and not proc.stderr.strip()


def probe_codec_shape(path: str | Path) -> tuple[str, int, int]:
    """Return (codec_name, height, width)."""
    info = ffprobe_streams(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return s.get("codec_name", "unknown"), int(s["height"]), int(s["width"])
    raise VideoError(f"No video stream in {path}")


def reencode_select_frames(
    src: str | Path,
    dst: str | Path,
    keep_indices: list[int],
    crop_ratio: Optional[tuple[float, float, float, float]] = None,
    resize_wh: Optional[tuple[int, int]] = None,
    codec: str = "libx264",
    crf: int = 18,
    pix_fmt: str = "yuv420p",
    fps: Optional[float] = None,
) -> None:
    """Re-encode ``src`` keeping only ``keep_indices`` (0-based frame numbers),
    optionally cropping by ratio and resizing, writing to ``dst``.

    The resulting video has exactly ``len(keep_indices)`` frames in the given
    order, so it stays positionally aligned with the cleaned parquet rows.

    crop_ratio is (x_ratio, y_ratio, w_ratio, h_ratio) relative to input WxH.
    """
    _require_ffmpeg()
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    filters = []

    # Frame selection: build a select expression matching the kept frame numbers.
    keep_set = set(int(i) for i in keep_indices)
    is_contiguous = keep_indices == list(range(keep_indices[0], keep_indices[-1] + 1)) if keep_indices else False
    if not (is_contiguous and keep_indices and keep_indices[0] == 0 and len(keep_indices) == _maybe_total(src)):
        # Need an explicit select filter. eq(n,i) terms — chunk to avoid huge exprs.
        expr = "+".join(f"eq(n\\,{i})" for i in sorted(keep_set))
        if not expr:
            raise VideoError("keep_indices is empty")
        filters.append(f"select='{expr}'")

    if crop_ratio is not None:
        x_r, y_r, w_r, h_r = crop_ratio
        # ffmpeg crop=w:h:x:y using input dimensions iw/ih.
        filters.append(
            f"crop=iw*{w_r}:ih*{h_r}:iw*{x_r}:ih*{y_r}"
        )
    if resize_wh is not None:
        w, h = resize_wh
        filters.append(f"scale={w}:{h}")

    vf = ",".join(filters) if filters else "null"

    cmd = [
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-vf", vf, "-vsync", "0",
        "-c:v", codec, "-crf", str(crf), "-pix_fmt", pix_fmt,
        "-an",
    ]
    if fps is not None:
        cmd += ["-r", str(fps)]
    cmd.append(str(dst))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoError(f"ffmpeg re-encode failed for {src}: {proc.stderr.strip()}")


def copy_or_reencode(
    src: str | Path,
    dst: str | Path,
    keep_indices: Optional[list[int]] = None,
    crop_ratio=None,
    resize_wh=None,
    **kwargs,
) -> None:
    """If no transform is needed, hard-link/copy; else re-encode."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    needs_transform = crop_ratio is not None or resize_wh is not None
    if keep_indices is not None:
        total = _maybe_total(src)
        if total is None or keep_indices != list(range(total)):
            needs_transform = True
    if not needs_transform:
        shutil.copy2(src, dst)
        return
    if keep_indices is None:
        total = _maybe_total(src) or 0
        keep_indices = list(range(total))
    reencode_select_frames(
        src, dst, keep_indices, crop_ratio=crop_ratio, resize_wh=resize_wh, **kwargs
    )


_total_cache: dict[str, Optional[int]] = {}


def _maybe_total(src: str | Path) -> Optional[int]:
    key = str(src)
    if key not in _total_cache:
        try:
            _total_cache[key] = count_frames(src)
        except VideoError:
            _total_cache[key] = None
    return _total_cache[key]
