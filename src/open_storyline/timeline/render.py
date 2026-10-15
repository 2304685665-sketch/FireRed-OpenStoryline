"""
Deterministic renderer for an EditPlan (ffmpeg + libass).

Layout (vertical 9:16): each shot is shown whole in the middle of the frame, black
letterbox/pillarbox bars cropped away automatically, over a blurred, darkened copy of
itself that fills the canvas. Text is drawn with ASS styles: a persistent header,
dish cards, punchline captions, dialogue captions, verdict stamps and an end card.

Audio: original audio per shot at its gain; background music under the whole video,
ducked during shots that carry speech; synthesized sound effects at overlay times;
loudness-normalized at the end.

    python -m open_storyline.timeline.render plan.json --media-root ~/Movies/eval_proxy \
        --fonts resource/fonts --out runs/food/draft.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from open_storyline.timeline.plan import EditPlan, Overlay, source_durations, validate

SFX = {  # name -> ffmpeg aevalsrc expression, duration
    "thud": ("0.9*sin(2*PI*70*t)*exp(-12*t)+0.35*sin(2*PI*140*t)*exp(-22*t)", 0.45),
    "ding": ("0.5*sin(2*PI*1320*t)*exp(-6*t)+0.25*sin(2*PI*1980*t)*exp(-8*t)", 0.7),
    "scratch": ("0.45*sin(2*PI*(900-1600*t)*t)*exp(-3*t)", 0.4),
    "pop": ("0.5*sin(2*PI*620*t)*exp(-28*t)", 0.18),
    "bleep": ("0.3*sin(2*PI*1000*t)", 0.6),
}

FONT_FAMILIES = {  # ASS style -> font family name (must exist in the fonts dir)
    "Header": "庞门正道标题体免费版",
    "Card": "标小智无界黑",
    "Punch": "标小智无界黑",
    "Dialogue": "Smiley Sans",
    "Stamp": "标小智无界黑",
    "End": "标小智无界黑",
}


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])} ... failed:\n{p.stderr[-2000:]}")
    return p.stderr


def probe_duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "default=noprint_wrappers=1:nokey=1", path], capture_output=True, text=True)
    return float(out.stdout.strip())


def has_audio(path: str) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", path], capture_output=True, text=True)
    return bool(out.stdout.strip())


def detect_crop(path: str, in_s: float, dur: float) -> str | None:
    """Crop that removes black bars in this window (ffmpeg cropdetect), or None."""
    err = run(["ffmpeg", "-hide_banner", "-ss", f"{in_s:.3f}", "-t", f"{min(dur, 3):.3f}", "-i", path,
               "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"])
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", err)
    if not found:
        return None
    w, h, x, y = map(int, found[-1])
    return f"{w}:{h}:{x}:{y}" if w > 16 and h > 16 else None


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_header(w: int, h: int) -> str:
    f = FONT_FAMILIES
    # Colours are &HAABBGGRR
    styles = [
        f"Style: Header,{f['Header']},86,&H0036D7FF,&H00000000,&H00000000,&H64000000,0,0,0,0,100,100,2,0,1,7,0,8,40,40,170,1",
        f"Style: Card,{f['Card']},62,&H00FFFFFF,&H00000000,&H00202020,&HC0202020,0,0,0,0,100,100,1,0,3,14,0,7,60,60,0,1",
        f"Style: Punch,{f['Punch']},78,&H0036D7FF,&H00000000,&H00000000,&H64000000,0,0,0,0,100,100,1,0,1,7,2,8,50,50,0,1",
        f"Style: Dialogue,{f['Dialogue']},62,&H00FFFFFF,&H00000000,&H00000000,&H64000000,0,0,0,0,100,100,1,0,1,5,1,8,60,60,0,1",
        f"Style: Stamp,{f['Stamp']},150,&H003030E0,&H00000000,&H00FFFFFF,&H00000000,0,0,0,0,100,100,6,0,1,9,0,5,0,0,0,1",
        f"Style: End,{f['End']},92,&H0036D7FF,&H00000000,&H00000000,&H78000000,0,0,0,0,100,100,2,0,1,8,3,5,60,60,0,1",
    ]
    return ("[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
            f"PlayResX: {w}\nPlayResY: {h}\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding\n" + "\n".join(styles) +
            "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")


def ass_event(o: Overlay, t0: float, t1: float, w: int, h: int) -> str:
    cx = w // 2
    if o.kind == "card":
        style, text = "Card", rf"{{\an7\pos(60,{int(h * 0.255)})\fad(120,120)}}{o.text}"
    elif o.kind == "punch":
        style, text = "Punch", rf"{{\an8\pos({cx},{int(h * 0.735)})\fscx70\fscy70\t(0,120,\fscx100\fscy100)}}{o.text}"
    elif o.kind == "dialogue":
        style, text = "Dialogue", rf"{{\an8\pos({cx},{int(h * 0.82)})}}{o.text}"
    elif o.kind == "stamp":
        colour = r"\1c&H003030E0&" if o.verdict == "red" else r"\1c&H00141414&"
        style, text = "Stamp", (rf"{{\an5\pos({int(w * 0.74)},{int(h * 0.36)}){colour}\frz14"
                                rf"\fscx220\fscy220\alpha&H40&\t(0,140,\fscx100\fscy100\alpha&H00&)}}{o.text}")
    elif o.kind == "endcard":
        style, text = "End", rf"{{\an5\pos({cx},{int(h * 0.5)})\fad(200,0)}}{o.text}"
    else:  # title
        style, text = "Punch", rf"{{\an5\pos({cx},{int(h * 0.5)})\fscx130\fscy130\fad(100,150)}}{o.text}"
    text = text.replace("\n", r"\N")
    return f"Dialogue: 2,{ass_time(t0)},{ass_time(t1)},{style},,0,0,0,,{text}"


def build_ass(plan: EditPlan) -> str:
    lines = [ass_header(plan.width, plan.height)]
    total = plan.duration
    if plan.header:
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(total)},Header,,0,0,0,,{plan.header}")
    for shot, t in zip(plan.shots, plan.starts()):
        for o in shot.overlays:
            end = shot.duration if o.end is None else o.end
            lines.append(ass_event(o, t + o.start, t + end, plan.width, plan.height))
    return "\n".join(lines) + "\n"


def render_shot(src: str, shot, plan: EditPlan, out: Path) -> None:
    w, h, fps = plan.width, plan.height, plan.fps
    crop = detect_crop(src, shot.in_s, shot.duration)
    fg = f"crop={crop}," if crop else ""
    # portrait sources fit by height, landscape by width
    fg += (f"scale='if(gt(ih/iw,1.2),-2,{w})':'if(gt(ih/iw,1.2),{int(h * 0.68)},-2)':flags=lanczos,setsar=1")
    vf = (f"[0:v]split=2[a][b];[a]{'crop=' + crop + ',' if crop else ''}scale={w}:{h}:force_original_aspect_ratio=increase,"
          f"crop={w}:{h},gblur=sigma=28,eq=brightness=-0.18:saturation=1.15[bg];[b]{fg}[fg];"
          f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={fps},format=yuv420p[v]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{shot.in_s:.3f}", "-t", f"{shot.duration:.3f}", "-i", src]
    if has_audio(src):
        af = f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume={shot.audio}[aud]"
    else:
        cmd += ["-f", "lavfi", "-t", f"{shot.duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        af = "[1:a]anull[aud]"
    cmd += ["-filter_complex", f"{vf};{af}", "-map", "[v]", "-map", "[aud]", "-c:v", "libx264", "-preset",
            "veryfast", "-crf", "19", "-c:a", "aac", "-b:a", "192k", "-t", f"{shot.duration:.3f}", str(out)]
    run(cmd)


def music_volume_expr(plan: EditPlan) -> str:
    parts = []
    for shot, t in zip(plan.shots, plan.starts()):
        g = plan.music_duck_gain if shot.audio >= 0.6 else plan.music_gain
        parts.append((t, t + shot.duration, g))
    expr = str(plan.music_gain)
    for a, b, g in reversed(parts):
        expr = f"if(between(t,{a:.3f},{b:.3f}),{g},{expr})"
    return expr


def render(plan: EditPlan, media_root: str | Path, fonts_dir: str | Path, out: str | Path,
           work: str | Path | None = None, music_root: str | Path | None = None) -> dict:
    media_root, out = Path(media_root).expanduser(), Path(out)
    work = Path(work or out.parent / (out.stem + "_work"))
    work.mkdir(parents=True, exist_ok=True)
    issues = validate(plan, source_durations(media_root, plan, probe_duration))
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise ValueError("plan has errors:\n" + "\n".join(f"- {i.where}: {i.message}" for i in errors))

    # fonts: libass takes one directory; collect font files into it
    fdir = work / "fonts"
    fdir.mkdir(exist_ok=True)
    for f in Path(fonts_dir).rglob("*"):
        if f.suffix.lower() in (".ttf", ".otf", ".ttc") and not (fdir / f.name).exists():
            shutil.copy(f, fdir / f.name)

    parts = []
    for i, shot in enumerate(plan.shots):
        p = work / f"shot_{i:03d}.mp4"
        render_shot(str(media_root / shot.source), shot, plan, p)
        parts.append(p)
    (work / "list.txt").write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    base = work / "base.mp4"
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "list.txt"), "-c", "copy", str(base)])
    ass = work / "captions.ass"
    ass.write_text(build_ass(plan), encoding="utf-8")

    total = plan.duration
    inputs = ["-i", str(base)]
    filters = [f"[0:v]subtitles='{ass}':fontsdir='{fdir}'[v]"]
    mix = ["[0:a]"]
    n = 1
    if plan.music:
        music = Path(music_root or ".") / plan.music
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        filters.append(f"[{n}:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo,"
                       f"volume='{music_volume_expr(plan)}':eval=frame,afade=t=in:d=0.6,"
                       f"afade=t=out:st={max(0, total - 1.5):.3f}:d=1.5[m]")
        mix.append("[m]")
        n += 1
    for shot, t in zip(plan.shots, plan.starts()):
        for o in shot.overlays:
            if o.sfx and o.sfx in SFX:
                expr, d = SFX[o.sfx]
                inputs += ["-f", "lavfi", "-i", f"aevalsrc='{expr}':d={d}:s=48000"]
                delay = int((t + o.start) * 1000)
                filters.append(f"[{n}:a]aformat=channel_layouts=stereo,adelay={delay}|{delay}[s{n}]")
                mix.append(f"[s{n}]")
                n += 1
    filters.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first,"
                   f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]")
    run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", "-t", f"{total:.3f}", str(out)])
    return {"out": str(out), "duration": total, "shots": len(plan.shots),
            "warnings": [f"{i.where}: {i.message}" for i in issues if i.level == "warning"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--music-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="validate only")
    args = ap.parse_args()
    plan = EditPlan.from_dict(json.loads(Path(args.plan).read_text(encoding="utf-8")))
    if args.check:
        issues = validate(plan, source_durations(Path(args.media_root).expanduser(), plan, probe_duration))
        for i in issues:
            print(f"[{i.level}] {i.where}: {i.message}")
        print(f"{len(plan.shots)} shots, {plan.duration:.1f}s, {sum(i.level == 'error' for i in issues)} errors")
        return
    print(json.dumps(render(plan, args.media_root, args.fonts, args.out, music_root=args.music_root),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
