import asyncio
import json
import re

from open_storyline.director.catalog import Catalog, Line
from open_storyline.director.graph import DirectorConfig, direct
from open_storyline.retrieval.fakes import FakeColourEncoder
from open_storyline.timeline.plan import EditPlan, validate

LINES = [Line("L0", "clips/rgb.mp4", 0.5, 2.0, "红色电话响了"), Line("L1", "clips/rgb.mp4", 4.0, 5.5, "咸到爆炸")]
OUTLINE = {"target_s": 6, "order": "chronological", "header": "测试", "beats": [
    {"id": "b1", "purpose": "hook", "query": "咸到爆炸", "target_s": 2, "punch": "踩雷了", "verdict": None},
    {"id": "b2", "purpose": "setup", "scene": "clips/rgb.mp4", "query": "红色的电话画面", "target_s": 2,
     "card": "第1顿", "verdict": "black"},
    {"id": "b3", "purpose": "ending", "scene": "walk/white.mp4", "query": "白色的走廊", "target_s": 2, "endcard": "评论区见", "verdict": None},
]}


class Msg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    async def ainvoke(self, msg):
        self.calls.append(msg)
        return Msg(self.fn(msg))


def selector(msg):
    beat = re.search(r'"id": "(b\d)"', msg).group(1)
    if beat == "b1":
        return json.dumps({"segment_id": "clips/rgb.mp4#001", "in_s": 3.5, "out_s": 6.0,
                           "dialogue": [{"line_id": "L1", "text": "咸到爆炸！"}]})
    if beat == "b2":
        if "被剪断" in msg:  # repaired after critic feedback
            return json.dumps({"segment_id": "clips/rgb.mp4#000", "in_s": 0.3, "out_s": 2.5,
                               "dialogue": [{"line_id": "L0", "text": "红色电话响了"}]})
        return json.dumps({"segment_id": "clips/rgb.mp4#000", "in_s": 1.0, "out_s": 2.5,
                           "dialogue": [{"line_id": "L0", "text": "红色电话响了"}]})
    return json.dumps({"segment_id": "walk/white.mp4#000", "in_s": 0.0, "out_s": 3.0, "dialogue": []})


def make(fake_index, **cfg):
    index, _ = fake_index
    cat = Catalog(index, LINES, sources=["clips/rgb.mp4", "walk/white.mp4"], encoder=FakeColourEncoder())
    d, s = FakeLLM(lambda m: json.dumps(OUTLINE, ensure_ascii=False)), FakeLLM(selector)
    out = asyncio.run(direct("剪一条 6 秒的测试视频", cat, d, s, DirectorConfig(**cfg)))
    return cat, d, s, out


def test_speech_channel_finds_line_only_in_audio(fake_index):
    index, _ = fake_index
    cat = Catalog(index, LINES, sources=["clips/rgb.mp4"], encoder=FakeColourEncoder())
    top = cat.search("咸到爆炸", k=1)[0]
    assert top["segment_id"] == "clips/rgb.mp4#001" and top["lines"][0]["text"] == "咸到爆炸"


def test_critic_repairs_only_the_failing_beat_and_plan_is_valid(fake_index):
    cat, d, s, out = make(fake_index)
    assert len(d.calls) == 1
    assert len(s.calls) == 4  # 3 beats + 1 repair of b2 only
    assert sum("被剪断" in m for m in s.calls) == 1
    plan = EditPlan.from_dict(out["plan"])
    assert [sh.beat for sh in plan.shots] == ["b1", "b2", "b3"]  # hook first, ending last
    assert abs(plan.duration - 6) <= 6 * 0.15
    seg_len = {"clips/rgb.mp4": 18.0, "walk/white.mp4": 14.0}
    assert [i for i in validate(plan, seg_len) if i.level == "error"] == []
    b2 = plan.shots[1]
    cap = [o for o in b2.overlays if o.kind == "dialogue"][0]
    assert b2.in_s <= 0.5 and cap.start >= 0 and cap.end <= b2.duration  # whole line kept
    assert any(o.kind == "stamp" and o.verdict == "black" for o in b2.overlays)
    assert out["issues"] == {}


def test_without_critic_the_cut_line_ships(fake_index):
    _, _, s, out = make(fake_index, critic=False)
    assert len(s.calls) == 3
    b2 = EditPlan.from_dict(out["plan"]).shots[1]
    assert b2.in_s == 1.0  # line L0 starts at 0.5: cut off


def test_long_speech_segments_split_into_clauses():
    from open_storyline.director.catalog import split_line

    parts = split_line(Line("L9", "a.mp4", 6.8, 15.7, "好吃吗？比较好吃多了，那吃个吧，不吃那个了。"))
    assert [p.text for p in parts] == ["好吃吗？", "比较好吃多了，", "那吃个吧，", "不吃那个了。"]
    assert parts[0].start == 6.8 and abs(parts[-1].end - 15.7) < 0.02 and parts[1].line_id == "L9.1"
    assert split_line(Line("L1", "a.mp4", 0, 2, "短句，不拆")) == [Line("L1", "a.mp4", 0, 2, "短句，不拆")]


def test_profanity_masked_in_captions():
    from open_storyline.director.graph import mask_profanity

    assert mask_profanity("你他妈这回给你填到爆") == "你××这回给你填到爆"


def test_beat_that_makes_no_progress_is_not_retried_forever(fake_index):
    index, _ = fake_index
    cat = Catalog(index, LINES, sources=["clips/rgb.mp4", "walk/white.mp4"], encoder=FakeColourEncoder())

    def stubborn(msg):  # always cuts line L0 in b2, ignoring feedback
        if '"id": "b2"' in msg:
            return json.dumps({"segment_id": "clips/rgb.mp4#000", "in_s": 1.0, "out_s": 2.5,
                               "dialogue": [{"line_id": "L0", "text": "红色电话响了"}]})
        return selector(msg)

    s = FakeLLM(stubborn)
    out = asyncio.run(direct("x", cat, FakeLLM(lambda m: json.dumps(OUTLINE, ensure_ascii=False)), s,
                             DirectorConfig(max_rounds=5)))
    assert len(s.calls) == 4  # one retry, then stopped: same issue again means no progress
    assert "b2" in out["issues"] and any("no progress" in x for x in out["log"])


def test_transcript_filter_drops_fragments_and_other_languages(tmp_path):
    from open_storyline.director.catalog import load_transcript

    f = tmp_path / "t.json"
    f.write_text(json.dumps([
        {"clip": "a", "start": 0, "end": 1, "lang": "<|zh|>", "text": "咸到爆炸"},
        {"clip": "a", "start": 1, "end": 2, "lang": "<|en|>", "text": "Yeah."},
        {"clip": "a", "start": 2, "end": 3, "lang": "<|zh|>", "text": "嗯。"},
        {"clip": "a", "start": 3, "end": 4, "lang": "<|ja|>", "text": "頑張れ大丈夫"},
    ], ensure_ascii=False), encoding="utf-8")
    assert [ln.text for ln in load_transcript(f)] == ["咸到爆炸"]


def test_fault_injection_runner_smoke(fake_index, monkeypatch, tmp_path):
    import types

    from benchmarks import fault_injection as fi
    from benchmarks import run_director

    index, _ = fake_index

    def fake_make_llm(cfg, writer, role, temperature, max_retries=2):
        return FakeLLM(lambda m: json.dumps(OUTLINE, ensure_ascii=False) if "分镜大纲" in m else selector(m))

    monkeypatch.setattr(run_director, "make_llm", fake_make_llm)
    monkeypatch.setattr(fi, "ROOT", tmp_path)
    args = types.SimpleNamespace(label="t", rate=0.6, seed=3, fallback_model=None)
    brief = {"id": "x1", "brief": "测试", "media": ["clips/rgb.mp4", "walk/white.mp4"]}
    rows = {c: asyncio.run(fi.run_config(brief, c, args, None, index, FakeColourEncoder(), LINES, {}))
            for c in ("none", "resilient")}
    assert rows["resilient"]["complete"]
    assert rows["resilient"]["calls_failed"] <= rows["none"]["calls_failed"]
    assert rows["none"]["injected"] or rows["resilient"]["injected"]


def test_scene_blocks_are_contiguous_and_chronological():
    from open_storyline.director.graph import order_beats, scene_splits

    beats = [{"id": "h", "purpose": "hook"}, {"id": "a1", "purpose": "setup"}, {"id": "b1", "purpose": "setup"},
             {"id": "a2", "purpose": "verdict"}, {"id": "c1", "purpose": "setup"}, {"id": "e", "purpose": "ending"}]
    src = {"h": "clip15.mp4", "a1": "clip5.mp4", "a2": "clip5.mp4", "b1": "clip22.mp4", "c1": "clip14.mp4",
           "e": "clip43.mp4"}
    out = [b["id"] for b in order_beats(beats, lambda b: src[b["id"]], {}, chronological=True)]
    assert out == ["h", "a1", "a2", "c1", "b1", "e"]  # clip5 block, clip14, clip22
    narrative = [b["id"] for b in order_beats(beats, lambda b: src[b["id"]], {}, chronological=False)]
    assert narrative == ["h", "a1", "a2", "b1", "c1", "e"]
    assert scene_splits(["x", "a", "b", "a", "y"]) == 1 and scene_splits(["x", "a", "a", "b", "y"]) == 0


def test_search_can_be_restricted_to_one_scene(fake_index):
    index, _ = fake_index
    cat = Catalog(index, LINES, sources=["clips/rgb.mp4", "walk/white.mp4"], encoder=FakeColourEncoder())
    got = cat.search("红色的电话画面", k=3, sources={"walk/white.mp4"})
    assert got and all(c["source"] == "walk/white.mp4" for c in got)


def test_facts_text_only_for_brief_scenes_and_marks_provenance():
    from open_storyline.director.catalog import facts_text

    facts = {"scenes": [
        {"scene": ["a.mp4"], "dish": "寿司", "verdict": "red",
         "facts": [{"text": "挺好吃", "source": "creator"}, {"text": "「放在我手上？」", "source": "audio"}]},
        {"scene": ["b.mp4"], "dish": "和牛", "verdict": "black", "facts": [{"text": "很腻", "source": "creator"}]},
        {"other_creator_facts": ["x"]}]}
    t = facts_text(facts, {"a.mp4"})
    assert "寿司" in t and "红榜" in t and "（创作者补充）" in t and "（原声）" in t and "和牛" not in t
