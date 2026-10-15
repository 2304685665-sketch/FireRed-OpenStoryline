"""
Attack suite for the render guard: malicious EditPlans vs legitimate ones.

A plan is model output, so a prompt-injected or confused model can hand the renderer
anything. Each attack plan is judged by an oracle of what the pipeline would actually do
WITHOUT the guard (validator + renderer only) and WITH it (check_plan first):

  path       source/music outside the allowed folders that exists -> the renderer reads it
  ass        caption text whose ASS override tags reach the subtitle file
  flood      caption text that injects more than two line breaks into the subtitle file
  resource   plan accepted with out-of-range numbers (NaN gains, 1000x volume, hour-long
             output, thousands of shots)

Legitimate plans (realistic captions with Chinese punctuation, emoji, numbers; normal
ranges) measure false blocks: rejected, or text changed.

    python benchmarks/attack_suite.py --out benchmarks/results/attack_suite.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_storyline.timeline.guard import PlanRejected, check_plan  # noqa: E402
from open_storyline.timeline.plan import EditPlan, Overlay, Shot, validate  # noqa: E402
from open_storyline.timeline.render import build_ass  # noqa: E402

LEGIT_TEXT = [
    "感觉像吃了十把盐", "哦，里面有葱", "我们点的：舒芙蕾", "第3顿 · 水果松饼", "和牛烤肉：此生不再吃",
    "“大家记住，这个特别难吃”", "价格 ¥1780（约 80 元）", "评分 2/10 😂", "日本美食红黑榜", "第一口：好香！",
    "你在日本踩过什么雷？\n评论区见", "「便利店夜宵」测评", "Day 3 · 大阪", "100% 会再来吗？不会", "(笑)",
    "#日本旅行 #美食", "这是 A5 和牛？", "拉面 vs 寿司", "结论：待你来写", "3 秒后反转——",
]
ASS_PAYLOADS = [r"{\fs400}巨大字", r"{\p1}m 0 0 l 1080 0 1080 1920 0 1920{\p0}", r"{\pos(9999,9999)}藏起来",
                r"{\alpha&HFF&}看不见", r"{\c&H0000FF&}{\fscx800}", r"正常{\k500}卡拉OK", r"{\move(0,0,1080,1920)}飞走",
                r"{\clip(0,0,1,1)}裁掉", r"{\fad(0,99999)}", r"{\r}{\bord99}"]


def base_plan(media: list[str]) -> EditPlan:
    return EditPlan(title="t", header="日本美食红黑榜", music="resource/bgms/music_0001.mp3", shots=[
        Shot(media[i % len(media)], 1.0 + i, 4.0 + i, 1.0, f"b{i}", [Overlay("punch", "好吃", 0.2, 2.5)])
        for i in range(4)])


def oracle_unguarded(plan: EditPlan, media_root: Path, music_root: Path, durations: dict) -> set[str]:
    """What gets through when only the validator and renderer stand in the way."""
    hits = set()
    errors = [i for i in validate(plan, {k: v for k, v in durations.items()}) if i.level == "error"]
    for s in plan.shots:
        real = (media_root / s.source).resolve()
        if real.is_file() and not real.is_relative_to(media_root.resolve()):
            hits.add("path")
    if plan.music:
        real = (music_root / plan.music).resolve()
        if real.is_file() and not real.is_relative_to((music_root / "resource" / "bgms").resolve()):
            hits.add("path")
    ass = build_ass(plan)
    for p in ASS_PAYLOADS:
        tag = p[p.index("{"):p.index("}") + 1] if "{" in p else p
        if tag in ass:
            hits.add("ass")
    if any(e.count(r"\N") > 2 for e in ass.splitlines() if e.startswith("Dialogue")):
        hits.add("flood")
    if not errors:
        nums = [plan.music_gain, plan.music_duck_gain] + [s.audio for s in plan.shots]
        if (any(not math.isfinite(x) or x < 0 or x > 2 for x in nums) or plan.duration > 300
                or len(plan.shots) > 80):
            hits.add("resource")
    return hits & {"path", "ass", "flood", "resource"}


def durations_for(plan: EditPlan, media_root: Path) -> dict:
    out = {}
    for s in plan.shots:
        if (media_root / s.source).exists():
            out[s.source] = 3600.0  # assume long files: range checks are not what is under test
    return out


def make_attacks(media: list[str], rng: random.Random) -> list[tuple[str, EditPlan]]:
    attacks: list[tuple[str, EditPlan]] = []
    paths = ["../secret/id_rsa.mp4", "../../secret/id_rsa.mp4", "./../secret/id_rsa.mp4", "sub/../../secret/id_rsa.mp4",
             "link_out.mp4", "{ABS}", "../secret/config.toml.mp4", "..%2fsecret/id_rsa.mp4"]
    for i in range(60):
        p = base_plan(media)
        target = paths[i % len(paths)]
        p.shots[rng.randrange(4)].source = target
        attacks.append(("path", p))
    for i in range(20):
        p = base_plan(media)
        p.music = ["../../../secret/id_rsa.mp3", "resource/bgms/../../secret/key.mp3", "{ABSMUSIC}",
                   "resource/../../secret/key.mp3"][i % 4]
        attacks.append(("path", p))
    for i in range(60):
        p = base_plan(media)
        payload = ASS_PAYLOADS[i % len(ASS_PAYLOADS)]
        o = rng.choice(p.shots).overlays[0]
        o.text = payload if i % 2 else f"好吃{payload}"
        if i % 3 == 0:
            p.header = payload
        attacks.append(("ass", p))
    for i in range(20):
        p = base_plan(media)
        p.shots[0].overlays[0].text = "刷屏\n" * (5 + i)
        attacks.append(("flood", p))
    for i in range(40):
        p = base_plan(media)
        kind = i % 5
        if kind == 0:
            p.shots[0].audio = float("nan")
        elif kind == 1:
            p.shots[1].audio = 1000.0
        elif kind == 2:
            p.music_gain = 500.0
        elif kind == 3:
            p.shots = [Shot(media[0], 0.0, 29.0, 1.0, "x", []) for _ in range(120)]
        else:
            p.shots[2].out_s = p.shots[2].in_s + 3000.0
        attacks.append(("resource", p))
    return attacks


def make_legit(media: list[str], rng: random.Random, n: int = 200) -> list[EditPlan]:
    out = []
    for _ in range(n):
        shots = []
        for j in range(rng.randint(4, 12)):
            a = round(rng.uniform(0, 50), 2)
            ovs = [Overlay("card", rng.choice(LEGIT_TEXT), 0.0, None, sfx="pop")]
            if rng.random() < 0.7:
                ovs.append(Overlay("dialogue", rng.choice(LEGIT_TEXT), 0.3, 2.0))
            if rng.random() < 0.3:
                ovs.append(Overlay("stamp", "黑榜", 1.0, None, verdict="black", sfx="thud"))
            shots.append(Shot(rng.choice(media), a, a + round(rng.uniform(1.0, 6.0), 2), rng.choice([0.3, 1.0]),
                              f"b{j}", ovs))
        out.append(EditPlan(title="t", header=rng.choice(["日本美食红黑榜", "", "Day 1 · 东京"]),
                            music=f"resource/bgms/music_000{rng.randint(1, 3)}.mp3", shots=shots))
    return out


def setup(tmp: Path) -> tuple[Path, Path, list[str]]:
    media_root, music_root = tmp / "media", tmp / "app"
    (music_root / "resource" / "bgms").mkdir(parents=True)
    media_root.mkdir()
    (tmp / "secret").mkdir()
    for f in ("id_rsa.mp4", "config.toml.mp4", "id_rsa.mp3", "key.mp3"):
        (tmp / "secret" / f).write_text("SECRET")
    media = [f"大致_clip{i}.mp4" for i in (5, 14, 22, 26, 33)]
    for m in media:
        (media_root / m).write_bytes(b"\0")
    for i in range(4):
        (music_root / "resource" / "bgms" / f"music_000{i}.mp3").write_bytes(b"\0")
    (media_root / "link_out.mp4").symlink_to(tmp / "secret" / "id_rsa.mp4")
    return media_root, music_root, media


def real_legit_plans(media: list[str]) -> list[EditPlan]:
    """Plans actually written (reference edit, director outputs), sources mapped onto the test media."""
    out = []
    files = list((ROOT / "benchmarks" / "references").glob("*.json")) + list((ROOT / "outputs").glob("*/plan.json"))
    for f in files:
        p = EditPlan.from_dict(json.loads(f.read_text(encoding="utf-8")))
        for i, s in enumerate(p.shots):
            s.source = media[i % len(media)]
            s.in_s, s.out_s = 1.0, 1.0 + min(s.duration, 29.0)
            for o in s.overlays:
                o.start = min(o.start, s.duration - 0.01)
                o.end = None if o.end is None else min(o.end, s.duration)
        p.music = "resource/bgms/music_0001.mp3"
        out.append(p)
    return out


def run(seed: int = 20261016) -> dict:
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        media_root, music_root, media = setup(tmp)
        attacks = make_attacks(media, rng)
        for _, p in attacks:  # absolute paths only known once the temp dir exists
            for s in p.shots:
                if s.source == "{ABS}":
                    s.source = str(tmp / "secret" / "id_rsa.mp4")
            if p.music == "{ABSMUSIC}":
                p.music = str(tmp / "secret" / "key.mp3")
        unguarded, guarded, cats = Counter(), Counter(), Counter()
        for cat, p in attacks:
            cats[cat] += 1
            if cat in oracle_unguarded(p, media_root, music_root, durations_for(p, media_root)):
                unguarded[cat] += 1
            g = copy.deepcopy(p)
            try:
                check_plan(g, media_root, music_root)
            except PlanRejected:
                continue
            if cat in oracle_unguarded(g, media_root, music_root, durations_for(g, media_root)):
                guarded[cat] += 1
        real = real_legit_plans(media)
        legit = make_legit(media, rng) + real
        blocked = altered = 0
        for p in legit:
            g = copy.deepcopy(p)
            try:
                check_plan(g, media_root, music_root)
            except PlanRejected:
                blocked += 1
                continue
            if g.to_dict() != p.to_dict():
                altered += 1
    return {"seed": seed, "attacks": sum(cats.values()), "by_category": dict(cats),
            "succeeded_without_guard": sum(unguarded.values()), "without_guard_by_category": dict(unguarded),
            "succeeded_with_guard": sum(guarded.values()), "with_guard_by_category": dict(guarded),
            "legit_plans": len(legit), "legit_real_plans": len(real), "legit_blocked": blocked, "legit_text_altered": altered}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.seed)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
