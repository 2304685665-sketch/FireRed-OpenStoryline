import json
from pathlib import Path

from open_storyline.timeline.plan import EditPlan, Overlay, Shot, validate
from open_storyline.timeline.render import build_ass, music_volume_expr

ROOT = Path(__file__).resolve().parents[1]


def plan(*shots, **kw):
    return EditPlan(title="t", shots=list(shots), **kw)


def test_reference_plan_is_valid_against_declared_sources():
    p = EditPlan.from_dict(json.loads((ROOT / "benchmarks/references/food_redblack_v1.json").read_text()))
    durations = {s.source: max(x.out_s for x in p.shots if x.source == s.source) + 1 for s in p.shots}
    assert [i for i in validate(p, durations) if i.level == "error"] == []
    assert p.shots[0].beat == "hook" and 50 < p.duration < 90


def test_errors_for_missing_source_bad_range_and_overlay_outside_shot():
    p = plan(Shot("a.mp4", 2, 1), Shot("b.mp4", 0, 1), Shot("a.mp4", 0, 5),
             Shot("a.mp4", 0, 2, overlays=[Overlay("punch", "x", 1.5, 3.0)]))
    msgs = [(i.level, i.message) for i in validate(p, {"a.mp4": 4.0})]
    assert ("error", "in/out must satisfy 0 <= in < out") in msgs
    assert ("error", "source file not found") in msgs
    assert any(lv == "error" and "beyond file length" in m for lv, m in msgs)
    assert any(lv == "error" and "outside the shot" in m for lv, m in msgs)


def test_craft_warnings_hook_reuse_reading_speed_and_stamp_verdict():
    p = plan(Shot("a.mp4", 0, 6, overlays=[Overlay("punch", "很长很长很长很长很长很长很长很长的字幕", 4.0, 5.0)]),
             Shot("a.mp4", 5, 6, overlays=[Overlay("stamp", "黑榜")]))
    msgs = " | ".join(i.message for i in validate(p, {"a.mp4": 10}))
    assert "first 3.0s" in msgs and "overlaps shot 0" in msgs and "chars/s" in msgs and "no verdict" in msgs


def test_ass_events_use_absolute_times_and_music_ducks_under_speech():
    p = plan(Shot("a.mp4", 0, 2, audio=0.2), Shot("a.mp4", 2, 5, audio=1.0,
             overlays=[Overlay("dialogue", "你好", 0.5, 1.5), Overlay("stamp", "黑榜", 1, None, verdict="black")]),
             header="H", music_gain=0.3, music_duck_gain=0.05)
    ass = build_ass(p)
    assert "Dialogue: 2,0:00:02.50,0:00:03.50,Dialogue" in ass
    assert "0:00:03.00,0:00:05.00,Stamp" in ass and "Header,,0,0,0,,H" in ass
    expr = music_volume_expr(p)
    assert "between(t,2.000,5.000),0.05" in expr and "between(t,0.000,2.000),0.3" in expr
