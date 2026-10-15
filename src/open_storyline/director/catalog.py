"""
Material catalog for the director: shots of the brief's source files with what is
said in them, and retrieval over them (visual + speech, fused with RRF).

Speech comes from a transcript (``runs/speech/transcript.json``: one record per
speech segment with file stem, start/end seconds and text). Each indexed shot gets
the text of the speech segments that overlap it, which feeds a BM25 "speech"
channel: lines like "咸到爆炸" exist only in the audio, so pure visual retrieval
cannot find them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from open_storyline.retrieval.index import SegmentIndex
from open_storyline.retrieval.retrieval import BM25Channel, SegmentRetriever, VisualChannel


@dataclass
class Line:
    line_id: str
    source: str
    start: float
    end: float
    text: str


_CLAUSE = re.compile(r"[^，。！？!?；;,]+[，。！？!?；;,]*")


def split_line(line: Line, max_s: float = 3.5) -> list[Line]:
    """
    Split a long speech segment into clause-level lines at punctuation, timing each clause
    proportionally to its length. VAD segments run up to ~20 s; a caption has to be fully
    inside its shot, so a 9 s segment cannot be used in a 4 s shot unless it is split.
    """
    if line.end - line.start <= max_s:
        return [line]
    parts = [m.group(0).strip() for m in _CLAUSE.finditer(line.text) if m.group(0).strip()]
    if len(parts) < 2:
        return [line]
    total = sum(len(x) for x in parts)
    out, t = [], line.start
    for j, text in enumerate(parts):
        d = (line.end - line.start) * len(text) / total
        out.append(Line(f"{line.line_id}.{j}", line.source, round(t, 2), round(t + d, 2), text))
        t += d
    return out


_CJK = re.compile(r"[\u4e00-\u9fff]")


def usable(record: dict, min_cjk: int = 3) -> bool:
    """Keep Chinese/Cantonese speech with at least a few characters; drop fragments such as
    "Yeah。" or one-character mis-recognitions that would make bad captions."""
    lang = str(record.get("lang", "")).strip("<|>")
    return lang in ("zh", "yue", "") and len(_CJK.findall(record.get("text", ""))) >= min_cjk


def load_transcript(path: str | Path, split: bool = True, min_cjk: int = 3) -> list[Line]:
    out = []
    for i, r in enumerate(json.loads(Path(path).read_text(encoding="utf-8"))):
        if not usable(r, min_cjk):
            continue
        src = r["clip"] if r["clip"].endswith(".mp4") else r["clip"] + ".mp4"
        ln = Line(f"L{i:04d}", src, float(r["start"]), float(r["end"]), r["text"].strip())
        out.extend(split_line(ln) if split else [ln])
    return out


class Catalog:
    def __init__(self, index: SegmentIndex, lines: list[Line], sources: Optional[list[str]] = None,
                 encoder=None, channels: tuple[str, ...] = ("visual", "speech")):
        self.index = index
        self.sources = set(sources) if sources else {s.asset_path for s in index.segments}
        self.segments = {s.segment_id: s for s in index.segments if s.asset_path in self.sources}
        self.lines = [ln for ln in lines if ln.source in self.sources]
        self.lines_by_id = {ln.line_id: ln for ln in self.lines}
        speech = {}
        for sid, s in self.segments.items():
            said = [ln.text for ln in self.lines_for(s.asset_path, s.start, s.end)]
            if said:
                speech[sid] = " ".join(said)
        if speech:
            index.set_text("speech", speech)
        chans = []
        if "visual" in channels and encoder is not None:
            chans.append(VisualChannel(index, encoder, "max"))
        if "speech" in channels and speech:
            chans.append(BM25Channel(index, "speech"))
        if not chans:
            raise ValueError("no usable retrieval channel")
        self.retriever = SegmentRetriever(index, chans)

    def lines_for(self, source: str, start: float, end: float, pad: float = 0.0) -> list[Line]:
        return [ln for ln in self.lines if ln.source == source and ln.start < end + pad and ln.end > start - pad]

    def search(self, query: str, k: int = 8) -> list[dict]:
        """Top-k shots of the allowed sources, each with its speech lines."""
        found = []
        for p in self.retriever.search(query, k=k * 6):
            if p.segment_id not in self.segments:
                continue
            found.append({
                "segment_id": p.segment_id, "source": p.asset_path,
                "start": round(p.start, 2), "end": round(p.end, 2),
                "lines": [{"line_id": ln.line_id, "start": ln.start, "end": ln.end, "text": ln.text}
                          for ln in self.lines_for(p.asset_path, p.start, p.end, pad=1.0)],
            })
            if len(found) == k:
                break
        return found

    def overview(self, max_lines: int = 80) -> str:
        """Compact listing for the director: files in natural order with a few of their lines."""
        from open_storyline.utils.chronology import natural_key

        rows = []
        per_file: dict[str, list[Line]] = {}
        for ln in self.lines:
            per_file.setdefault(ln.source, []).append(ln)
        budget = max_lines
        for src in sorted(self.sources, key=natural_key):
            segs = [s for s in self.segments.values() if s.asset_path == src]
            length = max((s.end for s in segs), default=0.0)
            said = per_file.get(src, [])[: max(1, min(4, budget))]
            budget -= len(said)
            quotes = " / ".join(f"{ln.start:.0f}s「{ln.text[:24]}」" for ln in said) if said else "（无对白）"
            rows.append(f"- {src}（{length:.0f}s，{len(segs)} 个镜头）：{quotes}")
        return "\n".join(rows)
