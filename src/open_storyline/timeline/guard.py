"""
Render guard: an EditPlan comes from a model, so treat it as untrusted input.

The renderer reads files, runs ffmpeg and burns ASS text. Without a guard a plan can
  * read any file the process can read (``source: "../../.ssh/id_rsa"``, absolute paths,
    symlinks out of the media folder), or a music file from anywhere;
  * inject ASS override tags through captions (``{\\fs400\\p1}...``: drawing commands,
    giant fonts, moving or hiding text) or flood the frame with line breaks;
  * ask for absurd work (NaN/negative/huge times, thousands of shots, hour-long output).

``check_plan`` rejects plans that reference anything outside the allowed roots or carry
non-finite / out-of-range numbers, and neutralises caption text (ASS control characters
replaced, length capped, control characters removed). Normal captions pass unchanged.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from open_storyline.timeline.plan import EditPlan

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
SFX_ALLOWED = {"thud", "ding", "scratch", "pop", "bleep", None}
_ASS_CTRL = {"{": "｛", "}": "｝", "\\": "＼"}


@dataclass
class Limits:
    max_shots: int = 80
    max_total_s: float = 300.0
    max_shot_s: float = 30.0
    max_text_chars: int = 40
    max_gain: float = 2.0


class PlanRejected(ValueError):
    pass


def _inside(root: Path, rel: str, exts: set[str]) -> Path:
    if not isinstance(rel, str) or not rel or "\x00" in rel:
        raise PlanRejected(f"bad path {rel!r}")
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise PlanRejected(f"path escapes its folder: {rel!r}")
    real = (root / p).resolve()
    if not real.is_relative_to(root.resolve()):
        raise PlanRejected(f"path resolves outside its folder: {rel!r}")
    if real.suffix.lower() not in exts:
        raise PlanRejected(f"unexpected file type: {rel!r}")
    if not real.is_file():
        raise PlanRejected(f"no such file: {rel!r}")
    return real


def clean_text(text: str, limit: int) -> str:
    if not isinstance(text, str):
        raise PlanRejected("overlay text must be a string")
    t = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == "\n")
    for k, v in _ASS_CTRL.items():
        t = t.replace(k, v)
    t = re.sub(r"\n{2,}", "\n", t)
    if t.count("\n") > 1:
        t = t.replace("\n", " ")
    return t[:limit]


def _finite(x, name: str, lo: float, hi: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= hi:
        raise PlanRejected(f"{name}={x!r} outside [{lo}, {hi}]")
    return float(x)


def check_plan(plan: EditPlan, media_root: str | Path, music_root: str | Path | None = None,
               limits: Limits = Limits()) -> list[str]:
    """Validate and sanitise in place. Raises PlanRejected; returns notes on neutralised text."""
    media_root = Path(media_root).expanduser()
    notes: list[str] = []
    if not (1 <= len(plan.shots) <= limits.max_shots):
        raise PlanRejected(f"{len(plan.shots)} shots (allowed 1-{limits.max_shots})")
    if plan.music is not None:
        if music_root is None:
            raise PlanRejected("music given but no music folder allowed")
        mp = Path(plan.music)
        if mp.parts[:2] != ("resource", "bgms") or len(mp.parts) != 3:
            raise PlanRejected(f"music must be resource/bgms/<file>: {plan.music!r}")
        _inside(Path(music_root) / "resource" / "bgms", mp.name, AUDIO_EXT)
    for k in ("music_gain", "music_duck_gain"):
        _finite(getattr(plan, k), k, 0.0, limits.max_gain)
    total = 0.0
    for i, s in enumerate(plan.shots):
        _inside(media_root, s.source, VIDEO_EXT)
        a = _finite(s.in_s, f"shot {i} in_s", 0.0, 86400.0)
        b = _finite(s.out_s, f"shot {i} out_s", 0.0, 86400.0)
        if not (a < b and b - a <= limits.max_shot_s):
            raise PlanRejected(f"shot {i}: in/out {a}-{b} (max {limits.max_shot_s}s)")
        _finite(s.audio, f"shot {i} audio", 0.0, limits.max_gain)
        total += b - a
        for o in s.overlays:
            _finite(o.start, f"shot {i} overlay start", 0.0, limits.max_shot_s)
            if o.end is not None:
                _finite(o.end, f"shot {i} overlay end", 0.0, limits.max_shot_s)
            if o.sfx not in SFX_ALLOWED:
                raise PlanRejected(f"shot {i}: unknown sound effect {o.sfx!r}")
            cleaned = clean_text(o.text, limits.max_text_chars)
            if cleaned != o.text:
                notes.append(f"shot {i}: neutralised overlay text")
                o.text = cleaned
    if total > limits.max_total_s:
        raise PlanRejected(f"total {total:.0f}s over {limits.max_total_s:.0f}s")
    if plan.header:
        plan.header = clean_text(plan.header, limits.max_text_chars)
    return notes
