"""
Recompute per-run numbers from saved traces and print one row per run.

Reads every ``outputs/<session>/run.json`` (optionally filtered by --label), re-derives
the summary from ``trace/events.jsonl`` with the current price table, finds rendered
videos, probes their durations with ffprobe and compares the final video against the
duration asked for in the brief. Writes a CSV next to the ledger.

    python benchmarks/summarize_runs.py --label baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.run_briefs import load_prices, produced_videos  # noqa: E402
from open_storyline.telemetry.meter import load_trace, summarize  # noqa: E402


def target_seconds(brief: str) -> float | None:
    m = re.search(r"(\d+)\s*秒", brief)
    return float(m.group(1)) if m else None


def duration(path: str) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=noprint_wrappers=1:nokey=1", path], capture_output=True, text=True, timeout=30)
        return round(float(out.stdout.strip()), 1)
    except Exception:
        return None


def collect(outputs: Path, label: str | None, prices) -> list[dict]:
    rows = []
    for rj in sorted(outputs.glob("*/run.json")):
        run = json.loads(rj.read_text(encoding="utf-8"))
        if label and run.get("label") != label:
            continue
        trace = rj.parent / "trace" / "events.jsonl"
        s = summarize(load_trace(trace), prices) if trace.exists() else {}
        videos = produced_videos(rj.parent)
        durs = [duration(v) for v in videos]
        renders = (s.get("tool_calls_by_name") or {})
        n_render = sum(v for k, v in renders.items() if k.endswith("render_video"))
        agent_in = (s.get("input_tokens_by_tool") or {}).get("(agent)", 0)
        total_in = sum((s.get("input_tokens_by_tool") or {}).values()) or 0
        rows.append({
            "label": run["label"], "brief_id": run["brief_id"], "session_id": run["session_id"],
            "error": run.get("error"), "turns": run.get("turns"),
            "model_calls": s.get("model_calls"), "llm_calls": (s.get("by_role") or {}).get("llm", {}).get("calls", 0),
            "vlm_calls": (s.get("by_role") or {}).get("vlm", {}).get("calls", 0),
            "tool_calls": s.get("tool_calls"), "renders": n_render,
            "input_tokens": total_in, "agent_input_share": round(agent_in / total_in, 3) if total_in else None,
            "cost_cny": s.get("cost_cny"), "wall_s": run.get("elapsed_s"),
            "target_s": target_seconds(run["brief"]), "final_video_s": durs[-1] if durs else None,
            "all_video_s": durs, "videos": len(videos),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--prices", default=str(ROOT / "benchmarks" / "prices.json"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = collect(Path(args.outputs), args.label, load_prices(Path(args.prices)))
    if not rows:
        sys.exit("no runs found")
    cols = ["brief_id", "model_calls", "llm_calls", "vlm_calls", "tool_calls", "renders", "input_tokens",
            "agent_input_share", "cost_cny", "wall_s", "target_s", "all_video_s", "error"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    out = Path(args.csv or ROOT / "benchmarks" / "results" / f"runs_{args.label or 'all'}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
