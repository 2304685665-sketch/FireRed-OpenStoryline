"""
EditPlan: an explicit, checkable description of a short video, and its validator.

The plan is what a director (a person today, an agent later) produces; rendering is
deterministic (see ``render.py``). Keeping the plan as data means every decision can
be inspected, diffed, validated before any expensive render, and compared with a
reference edit.

A plan is a list of shots, in output order. Each shot takes ``[in_s, out_s)`` of one
source file, says how loud its original audio is, and carries overlays drawn on top
of it (captions, dish cards, verdict stamps, sound effects). Overlay times are relative
to the start of the shot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

OverlayKind = Literal["dialogue", "punch", "card", "stamp", "title", "endcard"]


@dataclass
class Overlay:
    kind: OverlayKind
    text: str
    start: float = 0.0          # seconds from shot start
    end: Optional[float] = None  # None = until the end of the shot
    verdict: Optional[Literal["red", "black"]] = None  # stamps only
    sfx: Optional[str] = None    # sound effect name played at ``start``


@dataclass
class Shot:
    source: str        # file name relative to the plan's media root
    in_s: float
    out_s: float
    audio: float = 1.0  # gain of the original audio (0 = muted)
    beat: str = ""      # which part of the story this shot serves (for review / evaluation)
    overlays: list[Overlay] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.out_s - self.in_s, 3)


@dataclass
class EditPlan:
    title: str
    header: str = ""                # persistent series title drawn at the top of every frame
    width: int = 1080
    height: int = 1920
    fps: int = 30
    music: Optional[str] = None
    music_gain: float = 0.35        # music level under shots without speech
    music_duck_gain: float = 0.08   # music level while original audio carries speech
    shots: list[Shot] = field(default_factory=list)
    notes: str = ""

    @property
    def duration(self) -> float:
        return round(sum(s.duration for s in self.shots), 3)

    def starts(self) -> list[float]:
        t, out = 0.0, []
        for s in self.shots:
            out.append(round(t, 3))
            t += s.duration
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EditPlan":
        shots = [Shot(**{**s, "overlays": [Overlay(**o) for o in s.get("overlays", [])]}) for s in d.get("shots", [])]
        return cls(**{**d, "shots": shots})


@dataclass
class Issue:
    level: Literal["error", "warning"]
    where: str
    message: str


def validate(plan: EditPlan, durations: dict[str, float], *, hook_max_s: float = 3.0,
             min_shot_s: float = 0.5, max_shot_s: float = 8.0, max_chars_per_s: float = 6.0) -> list[Issue]:
    """
    Check a plan before rendering. ``durations`` maps source file -> length in seconds.

    Errors make the plan unrenderable (missing source, range outside the file, empty or
    inverted shot, overlay outside its shot). Warnings flag editing-craft problems: a
    hook that takes too long to land, very short/long shots, captions too fast to read,
    a source reused for overlapping ranges, stamps without a verdict.
    """
    issues: list[Issue] = []
    if not plan.shots:
        return [Issue("error", "plan", "no shots")]
    used: dict[str, list[tuple[float, float, int]]] = {}
    for i, s in enumerate(plan.shots):
        where = f"shot {i} ({s.source} {s.in_s:.1f}-{s.out_s:.1f})"
        if s.source not in durations:
            issues.append(Issue("error", where, "source file not found"))
            continue
        if not (0 <= s.in_s < s.out_s):
            issues.append(Issue("error", where, "in/out must satisfy 0 <= in < out"))
            continue
        if s.out_s > durations[s.source] + 0.05:
            issues.append(Issue("error", where, f"out {s.out_s:.2f}s beyond file length {durations[s.source]:.2f}s"))
        if s.duration < min_shot_s:
            issues.append(Issue("warning", where, f"shot shorter than {min_shot_s}s"))
        if s.duration > max_shot_s:
            issues.append(Issue("warning", where, f"shot longer than {max_shot_s}s; consider cutting"))
        for a, b, j in used.get(s.source, []):
            if s.in_s < b and a < s.out_s:
                issues.append(Issue("warning", where, f"overlaps shot {j} of the same file"))
        used.setdefault(s.source, []).append((s.in_s, s.out_s, i))
        for o in s.overlays:
            end = s.duration if o.end is None else o.end
            if not (0 <= o.start < end <= s.duration + 1e-6):
                issues.append(Issue("error", where, f"{o.kind} overlay '{o.text}' outside the shot"))
                continue
            if o.kind == "stamp" and o.verdict not in ("red", "black"):
                issues.append(Issue("warning", where, f"stamp '{o.text}' has no verdict"))
            if o.kind in ("dialogue", "punch") and len(o.text) / max(end - o.start, 1e-6) > max_chars_per_s:
                issues.append(Issue("warning", where, f"caption '{o.text}' faster than {max_chars_per_s} chars/s"))
    first = plan.shots[0]
    hook = [o for o in first.overlays if o.kind in ("dialogue", "punch")]
    if not hook or min(o.start for o in hook) > hook_max_s:
        issues.append(Issue("warning", "hook", f"no caption or line lands within the first {hook_max_s}s"))
    return issues


def source_durations(root: str | Path, plan: EditPlan, probe) -> dict[str, float]:
    out = {}
    for name in {s.source for s in plan.shots}:
        p = Path(root) / name
        if p.exists():
            out[name] = probe(str(p))
    return out
