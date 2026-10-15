"""
Chronological ordering of clips by when the source footage was shot.

A clip's time key is (capture time of its source file, file name in natural order,
start offset inside the file). Capture time comes from container metadata
(``com.apple.quicktime.creationdate`` first: it is the shooting time on iPhone and
many cameras; then ``creation_time``). Files without readable metadata sort after
those with it, by natural file name (``clip2`` before ``clip10``).

Caveat: re-exported files often carry the *export* time in ``creation_time``. That is
still monotonic when files were exported in shooting order, but not in general, so
``source="filename"`` lets the caller ignore metadata when names encode the order.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

_INF = float("inf")


def natural_key(name: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(name).name))


def _parse_time(value: str) -> Optional[float]:
    v = value.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.fromisoformat(v) if fmt is None else datetime.strptime(v, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def probe_capture_time(path: str) -> Optional[float]:
    """Shooting time (POSIX seconds) from container tags, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(out.stdout or "{}")
    except Exception:
        return None
    tags: dict[str, str] = {}
    for block in [data.get("format", {})] + data.get("streams", []):
        for k, v in (block.get("tags") or {}).items():
            tags.setdefault(k.lower(), v)
    for key in ("com.apple.quicktime.creationdate", "creation_time"):
        if key in tags:
            t = _parse_time(tags[key])
            if t is not None:
                return t
    return None


def clip_time_keys(
    media: Iterable[dict[str, Any]],
    clips: Iterable[dict[str, Any]],
    source: str = "auto",
    probe: Callable[[str], Optional[float]] = probe_capture_time,
) -> dict[str, tuple]:
    """
    Map clip_id -> sortable time key.

    media: load_media output items ({"media_id", "path"}); clips: split_shots clips
    ({"clip_id", "source_ref": {"media_id", "start"}}). source: "auto"/"metadata" use
    capture time then file name; "filename" uses file name only.
    """
    media_key: dict[str, tuple] = {}
    for m in media:
        path = str(m.get("path", ""))
        t = probe(path) if source in ("auto", "metadata") and path else None
        media_key[m.get("media_id", "")] = (t if t is not None else _INF, natural_key(path))
    keys: dict[str, tuple] = {}
    for c in clips:
        ref = c.get("source_ref") or {}
        mk = media_key.get(ref.get("media_id", ""), (_INF, ()))
        keys[c.get("clip_id", "")] = (mk[0], mk[1], float(ref.get("start") or 0))
    return keys


def reorder_chronologically(groups: list[dict[str, Any]], keys: dict[str, tuple]) -> list[dict[str, Any]]:
    """Sort clips inside each group, then groups by their earliest clip; renumber group ids."""
    last = (_INF, (), _INF)
    out = []
    for g in groups:
        ids = sorted(g.get("clip_ids", []), key=lambda cid: keys.get(cid, last))
        out.append({**g, "clip_ids": ids})
    out.sort(key=lambda g: keys.get(g["clip_ids"][0], last) if g["clip_ids"] else last)
    for i, g in enumerate(out, start=1):
        g["group_id"] = f"group_{i:04d}"
    return out
