import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from benchmarks.attack_suite import run  # noqa: E402
from open_storyline.timeline.guard import PlanRejected, check_plan, clean_text  # noqa: E402
from open_storyline.timeline.plan import EditPlan, Overlay, Shot  # noqa: E402


def test_attack_suite_guard_blocks_all_and_no_false_blocks():
    r = run()
    assert r["attacks"] == 200 and r["succeeded_without_guard"] > 100
    assert r["succeeded_with_guard"] == 0
    assert r["legit_blocked"] == 0 and r["legit_text_altered"] == 0


def test_guard_rules(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.mp4").write_bytes(b"0")
    ok = EditPlan(title="t", shots=[Shot("a.mp4", 0, 2, overlays=[Overlay("punch", "好吃{\\fs300}", 0, 1)])])
    notes = check_plan(ok, tmp_path / "m")
    assert ok.shots[0].overlays[0].text == "好吃｛＼fs300｝" and notes
    for bad in ("../x.mp4", "/etc/passwd", "a.txt"):
        with pytest.raises(PlanRejected):
            check_plan(EditPlan(title="t", shots=[Shot(bad, 0, 2)]), tmp_path / "m")
    with pytest.raises(PlanRejected):
        check_plan(EditPlan(title="t", shots=[Shot("a.mp4", 0, 2, audio=float("nan"))]), tmp_path / "m")
    assert clean_text("你在日本踩过什么雷？\n评论区见", 40) == "你在日本踩过什么雷？\n评论区见"
