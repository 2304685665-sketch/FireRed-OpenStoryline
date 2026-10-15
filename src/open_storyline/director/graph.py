"""
Multi-agent editing as a LangGraph: director -> parallel shot selectors -> assembler
-> critic -> (targeted repair of failing beats) -> EditPlan.

* Director (LLM): reads the brief and a compact catalog (files, lengths, what is said
  where) and writes an outline: ordered beats, each with a retrieval query, a target
  length, and on-screen text. It does not pick shots.
* Selector (LLM, one per beat, run in parallel with Send): gets the beat and its top-k
  retrieved shots (visual + speech channels, RRF) with their transcript lines, and picks
  one shot, an in/out range, and the lines to caption.
* Assembler (code): orders beats (hook first; chronological if asked), turns selections
  into an EditPlan with cards, punchlines, stamps, captions and sound effects.
* Critic (code): checks every selection and the plan (validator, duration vs target,
  captions cut mid-line, captions not grounded in the transcript, a shot reused across
  beats). Duration is fixed deterministically first; beats that still fail are sent
  back to their selector with the issue text, up to ``max_rounds``. Only those beats
  are re-selected.

Rendering happens outside the graph (``timeline.render``), once, on a validated plan.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from open_storyline.director.catalog import Catalog
from open_storyline.timeline.plan import EditPlan, Overlay, Shot, validate
from open_storyline.utils.chronology import natural_key

DIRECTOR_PROMPT = """你是短视频导演。根据创作需求和素材目录，写出分镜大纲（只规划，不挑具体镜头）。

创作需求：
{brief}

素材目录（文件、时长、部分原声台词）：
{overview}

要求：
- 总时长贴合需求里的时长；没写时长就控制在 45 秒左右。
- 第一个 beat 必须是 purpose="hook"：3 秒内抛出全片最有冲突或最好笑的一句/一幕。
- 每个 beat 写一个 query：描述要找的画面或台词（会用于画面检索和原声检索）。
- 好笑的点优先用素材里的原声台词，不要编造台词。
- verdict（红榜 red / 黑榜 black）只有在需求明确说了好坏时才填，否则为 null。
- 需要按时间顺序时 order="chronological"，否则 "narrative"。

只输出 JSON：
{{"target_s": 数字, "order": "chronological|narrative", "header": "顶部标题（可空）",
  "beats": [{{"id": "b1", "purpose": "hook|setup|reaction|verdict|montage|ending", "query": "...",
              "target_s": 数字, "card": "左上角卡片（可空）", "punch": "大字吐槽（可空）",
              "verdict": "red|black|null", "endcard": "结尾互动语（仅 ending，可空）"}}]}}"""

SELECTOR_PROMPT = """你是剪辑师，为一个分镜挑镜头。

分镜：{beat}

候选镜头（检索得到，含原声台词与时间，单位秒，时间都是源文件内的绝对时间）：
{candidates}
{feedback}
要求：
- 只能从候选里选一个 segment_id；in_s/out_s 必须在该镜头 start-2 到 end+2 之间，时长接近分镜 target_s。
- 如果要用台词做字幕：只能引用候选里的 line_id，并且 in_s/out_s 要完整覆盖这句台词的时间；
  text 可以修正识别错别字，但不能改意思、不能编造。
- 没有合适台词就让 dialogue 为空。

只输出 JSON：{{"segment_id": "...", "in_s": 数字, "out_s": 数字,
  "dialogue": [{{"line_id": "...", "text": "..."}}], "reason": "一句话说明"}}"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(m.group(0))


def _merge(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}


def _add(a: list, b: list) -> list:
    return (a or []) + (b or [])


class State(TypedDict, total=False):
    brief: str
    outline: dict
    candidates: Annotated[dict, _merge]   # beat_id -> candidate list
    selections: Annotated[dict, _merge]   # beat_id -> selection
    issues: dict                          # beat_id -> [issue text]
    round: int
    plan: dict
    log: Annotated[list, _add]


@dataclass
class DirectorConfig:
    k: int = 8
    max_rounds: int = 2
    duration_tolerance: float = 0.15
    min_shot_s: float = 1.0
    music: Optional[str] = None
    critic: bool = True   # ablation switch: False = assemble once, no checks/repair


def _text_overlap(clean: str, raw: str) -> float:
    a, b = set(re.sub(r"\W", "", clean)), set(re.sub(r"\W", "", raw))
    return len(a & b) / max(1, len(a))


def build_graph(catalog: Catalog, director_llm, selector_llm, cfg: DirectorConfig = DirectorConfig()):
    seg_len = {}
    for s in catalog.segments.values():
        seg_len[s.asset_path] = max(seg_len.get(s.asset_path, 0.0), s.end)

    async def director(state: State) -> dict:
        msg = DIRECTOR_PROMPT.format(brief=state["brief"], overview=catalog.overview())
        outline = parse_json((await director_llm.ainvoke(msg)).content)
        beats = outline.get("beats") or []
        if not beats:
            raise ValueError("director produced no beats")
        for i, b in enumerate(beats):
            b["id"] = b.get("id") or f"b{i + 1}"
            if b.get("verdict") not in ("red", "black"):
                b["verdict"] = None
        return {"outline": outline, "round": 0, "issues": {}, "log": [f"director: {len(beats)} beats"]}

    def fan_out(state: State):
        beats = state["outline"]["beats"]
        todo = [b for b in beats if b["id"] in state.get("issues", {})] if state.get("round") else beats
        return [Send("select", {"beat": b, "feedback": state.get("issues", {}).get(b["id"], []),
                                "taken": [s["segment_id"] for bid, s in state.get("selections", {}).items()
                                          if bid != b["id"]]}) for b in todo]

    async def select(payload: dict) -> dict:
        beat = payload["beat"]
        cands = [c for c in catalog.search(beat.get("query") or beat.get("punch") or "", k=cfg.k + 4)
                 if c["segment_id"] not in payload.get("taken", [])][: cfg.k]
        fb = ""
        if payload.get("feedback"):
            fb = "\n上一次的选择没有通过检查，请修正：\n" + "\n".join(f"- {x}" for x in payload["feedback"]) + "\n"
        msg = SELECTOR_PROMPT.format(beat=json.dumps(beat, ensure_ascii=False),
                                     candidates=json.dumps(cands, ensure_ascii=False), feedback=fb)
        try:
            sel = parse_json((await selector_llm.ainvoke(msg)).content)
        except Exception as e:  # keep the graph going; the critic reports the beat
            sel = {"error": f"{type(e).__name__}: {e}"}
        return {"selections": {beat["id"]: sel}, "candidates": {beat["id"]: cands},
                "log": [f"select {beat['id']}: {sel.get('segment_id', sel.get('error'))}"]}

    def check_selection(beat: dict, sel: dict, cands: list[dict]) -> list[str]:
        if "error" in sel:
            return [f"模型输出无法解析：{sel['error'][:120]}"]
        by_id = {c["segment_id"]: c for c in cands}
        c = by_id.get(sel.get("segment_id"))
        if c is None:
            return ["segment_id 不在候选列表里"]
        try:
            a, b = float(sel["in_s"]), float(sel["out_s"])
        except (KeyError, TypeError, ValueError):
            return ["缺少 in_s/out_s"]
        out = []
        lo, hi = max(0.0, c["start"] - 2), min(seg_len.get(c["source"], c["end"]), c["end"] + 2)
        if not (lo <= a < b <= hi + 1e-6):
            out.append(f"in/out 必须在 {lo:.1f}-{hi:.1f} 之间且 in<out")
        if b - a < cfg.min_shot_s:
            out.append(f"镜头太短（{b - a:.1f}s），至少 {cfg.min_shot_s}s")
        for d in sel.get("dialogue") or []:
            ln = catalog.lines_by_id.get(d.get("line_id"))
            if ln is None or ln.source != c["source"]:
                out.append(f"字幕引用了不存在或不属于该镜头的台词 {d.get('line_id')}")
                continue
            if ln.start < a - 0.05 or ln.end > b + 0.05:
                out.append(f"台词 {ln.line_id}（{ln.start:.1f}-{ln.end:.1f}s）被剪断，in/out 需完整覆盖")
            if _text_overlap(d.get("text", ""), ln.text) < 0.4:
                out.append(f"字幕“{d.get('text', '')}”与原声“{ln.text}”出入太大，不能改意思")
        return out

    def assemble(state: State) -> dict:
        outline, sels = state["outline"], state.get("selections", {})
        beats = [b for b in outline["beats"] if "segment_id" in sels.get(b["id"], {})]
        cands = state.get("candidates", {})

        def src_of(b):
            return next((c["source"] for c in cands.get(b["id"], []) if c["segment_id"] == sels[b["id"]]["segment_id"]),
                        "")

        if outline.get("order") == "chronological":
            hook = [b for b in beats if b.get("purpose") == "hook"][:1]
            ending = [b for b in beats if b.get("purpose") == "ending" and b not in hook][-1:]
            middle = [b for b in beats if b not in hook and b not in ending]
            middle.sort(key=lambda b: (natural_key(src_of(b)), float(sels[b["id"]].get("in_s", 0))))
            beats = hook + middle + ending
        shots = []
        for b in beats:
            sel = sels[b["id"]]
            try:
                a, e = float(sel["in_s"]), float(sel["out_s"])
            except (KeyError, TypeError, ValueError):
                continue
            dur = max(0.01, e - a)
            ovs = []
            if b.get("card"):
                ovs.append(Overlay("card", b["card"], 0.0, None, sfx="pop"))
            if b.get("punch"):
                ovs.append(Overlay("punch", b["punch"], min(0.3, dur / 3), None))
            for d in sel.get("dialogue") or []:
                ln = catalog.lines_by_id.get(d.get("line_id"))
                if ln:
                    s0, s1 = max(0.0, ln.start - a), min(dur, ln.end - a)
                    if s1 > s0:
                        ovs.append(Overlay("dialogue", f"“{d.get('text') or ln.text}”", round(s0, 2), round(s1, 2)))
            if b.get("verdict"):
                ovs.append(Overlay("stamp", "红榜" if b["verdict"] == "red" else "黑榜", round(dur * 0.4, 2), None,
                                   verdict=b["verdict"], sfx="thud"))
            if b.get("endcard"):
                ovs.append(Overlay("endcard", b["endcard"], round(dur * 0.5, 2), None, sfx="ding"))
            speech = bool(sel.get("dialogue"))
            shots.append(Shot(src_of(b), round(a, 2), round(e, 2), 1.0 if speech else 0.3, b["id"], ovs))
        plan = EditPlan(title=outline.get("header") or "draft", header=outline.get("header") or "",
                        music=cfg.music, shots=shots, notes=json.dumps({"target_s": outline.get("target_s")}))
        return {"plan": plan.to_dict()}

    def fit_duration(plan: EditPlan, target: float) -> list[str]:
        """Deterministic first repair: trim or extend shots proportionally, never cutting a captioned line."""
        notes = []
        for _ in range(3):
            total = plan.duration
            if not target or abs(total - target) <= target * cfg.duration_tolerance:
                break
            scale = target / total
            for s in plan.shots:
                lines = [o for o in s.overlays if o.kind == "dialogue"]
                keep_from = min((o.start for o in lines), default=None)
                keep_to = max((o.end for o in lines), default=None)
                want = max(cfg.min_shot_s, s.duration * scale)
                if keep_from is not None:
                    want = max(want, keep_to - keep_from + 0.3)
                    a = max(0.0, s.in_s + keep_from - 0.15)
                else:
                    a = s.in_s
                limit = seg_len.get(s.source, s.out_s)
                e = min(limit, a + want)
                shift = a - s.in_s
                s.in_s, s.out_s = round(a, 2), round(e, 2)
                for o in s.overlays:
                    o.start = max(0.0, round(o.start - shift, 2))
                    if o.end is not None:
                        o.end = min(s.duration, round(o.end - shift, 2))
                        if o.end <= o.start:
                            o.end = None
            notes.append(f"duration {total:.1f}s -> {plan.duration:.1f}s (target {target:.0f}s)")
        return notes

    def critic(state: State) -> dict:
        outline, sels, cands = state["outline"], state.get("selections", {}), state.get("candidates", {})
        issues: dict[str, list[str]] = {}
        seen: dict[str, str] = {}
        for b in outline["beats"]:
            sel = sels.get(b["id"], {"error": "no selection"})
            errs = check_selection(b, sel, cands.get(b["id"], []))
            sid = sel.get("segment_id")
            if sid and sid in seen:
                errs.append(f"与 {seen[sid]} 用了同一个镜头")
            if sid:
                seen.setdefault(sid, b["id"])
            if errs:
                issues[b["id"]] = errs
        plan = EditPlan.from_dict(state["plan"])
        notes = fit_duration(plan, float(outline.get("target_s") or 0))
        for i in validate(plan, seg_len):
            if i.level == "error" and i.where.startswith("shot "):
                idx = int(i.where.split()[1])
                issues.setdefault(plan.shots[idx].beat, []).append(i.message)
        return {"issues": issues, "plan": plan.to_dict(), "round": state.get("round", 0) + 1,
                "log": notes + [f"critic round {state.get('round', 0) + 1}: {len(issues)} beats failing"]}

    def route(state: State):
        if state["issues"] and state["round"] <= cfg.max_rounds:
            return fan_out(state)
        return END

    g = StateGraph(State)
    g.add_node("director", director)
    g.add_node("select", select)
    g.add_node("assemble", assemble)
    g.add_edge(START, "director")
    g.add_conditional_edges("director", fan_out, ["select"])
    g.add_edge("select", "assemble")
    if cfg.critic:
        g.add_node("critic", critic)
        g.add_edge("assemble", "critic")
        g.add_conditional_edges("critic", route, ["select", END])
    else:
        g.add_edge("assemble", END)
    return g.compile()


async def direct(brief: str, catalog: Catalog, director_llm, selector_llm,
                 cfg: DirectorConfig = DirectorConfig()) -> dict[str, Any]:
    graph = build_graph(catalog, director_llm, selector_llm, cfg)
    return await graph.ainvoke({"brief": brief}, {"recursion_limit": 50})
