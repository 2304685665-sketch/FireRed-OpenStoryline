"""
Run editing briefs through the multi-agent director (director -> parallel selectors ->
critic -> targeted repair) and render each validated plan once.

Same briefs, same media, same trace format and ledger as run_briefs.py, so
summarize_runs.py compares the two systems row by row.

    python benchmarks/run_director.py --briefs benchmarks/briefs/japan_dev.jsonl \
        --label director --budget-cny 10 --limit 1
    python benchmarks/run_director.py ... --label director-nocritic --no-critic   # ablation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.run_briefs import BudgetExceeded, git_commit, load_prices, read_briefs  # noqa: E402


def make_llm(cfg, writer, role: str, temperature: float):
    from langchain_openai import ChatOpenAI

    from open_storyline.telemetry.meter import instrument

    llm = ChatOpenAI(model=cfg.llm.model, base_url=cfg.llm.base_url.rstrip("/"), api_key=cfg.llm.api_key,
                     timeout=60, temperature=temperature, max_retries=2)
    return instrument(llm, writer, role)


async def run_one(brief, args, prices, spent_before, encoder, index, lines) -> dict:
    from open_storyline.config import default_config_path, load_settings
    from open_storyline.director.catalog import Catalog
    from open_storyline.director.graph import DirectorConfig, direct
    from open_storyline.telemetry.meter import TraceWriter, load_trace, summarize
    from open_storyline.timeline.plan import EditPlan
    from open_storyline.timeline.render import render

    cfg = load_settings(default_config_path())
    session_id = f"{args.label}_{brief['id']}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(cfg.project.outputs_dir) / session_id
    trace_path = run_dir / "trace" / "events.jsonl"

    def budget_check() -> None:
        s = summarize(load_trace(trace_path), prices) if trace_path.exists() else {"cost_cny": 0.0}
        if (s["cost_cny"] or 0.0) + spent_before >= args.budget_cny:
            raise BudgetExceeded(f"budget {args.budget_cny} CNY reached")

    writer = TraceWriter(trace_path, guard=budget_check)
    sources = None if brief.get("media", "all") == "all" else list(brief["media"])
    catalog = Catalog(index, lines, sources=sources, encoder=encoder,
                      channels=tuple(args.channels))
    t0 = time.time()
    out, error, video = {}, None, None
    try:
        out = await direct(brief["brief"], catalog, make_llm(cfg, writer, "llm", 0.4),
                           make_llm(cfg, writer, "llm", 0.2),
                           DirectorConfig(critic=not args.no_critic, music=args.music))
        plan = EditPlan.from_dict(out["plan"])
        (run_dir / "plan.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        t_r = time.time()
        res = render(plan, args.media_root, ROOT / "resource" / "fonts", run_dir / "final.mp4",
                     work=ROOT / "runs" / "render_cache", music_root=ROOT)
        writer.write({"kind": "tool_call", "tool": "render_video", "latency_s": round(time.time() - t_r, 3),
                      "ok": True, "error": None, "t_start": t_r, "ts": time.time()})
        video = res["out"]
    except BudgetExceeded as e:
        error = f"BudgetExceeded: {e}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    summary = summarize(load_trace(trace_path), prices) if trace_path.exists() else {}
    record = {
        "label": args.label, "brief_id": brief["id"], "brief": brief["brief"], "session_id": session_id,
        "media": sources or "all", "commit": git_commit(),
        "models": {"llm": cfg.llm.model}, "channels": args.channels, "critic": not args.no_critic,
        "turns": 1, "auto_replies": 0, "error": error, "videos": [video] if video else [],
        "elapsed_s": round(time.time() - t0, 1), "summary": summary,
        "unresolved_issues": out.get("issues"), "repair_rounds": out.get("round"), "log": out.get("log"),
        "final_text": "",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


async def main_async(args) -> None:
    from open_storyline.director.catalog import load_transcript
    from open_storyline.retrieval.encoders import ChineseClipEncoder
    from open_storyline.retrieval.index import SegmentIndex

    from open_storyline.config import default_config_path, load_settings

    cfg = load_settings(default_config_path())
    if not (cfg.llm.api_key and cfg.llm.base_url and cfg.llm.model):
        sys.exit("LLM not configured: set OPENSTORYLINE_LLM_MODEL / _BASE_URL / _API_KEY in this terminal first")
    prices = load_prices(Path(args.prices))
    index = SegmentIndex.load(args.index)
    encoder = ChineseClipEncoder(args.model_dir) if "visual" in args.channels else None
    lines = load_transcript(args.transcript)
    briefs = read_briefs(Path(args.briefs))[: args.limit or None]
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for b in briefs:
        if spent >= args.budget_cny:
            print(f"budget reached ({spent:.2f} CNY); stopping before {b['id']}")
            break
        rec = await run_one(b, args, prices, spent, encoder, index, lines)
        cost = (rec["summary"] or {}).get("cost_cny")
        spent += cost or 0.0
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: rec[k] for k in ("label", "brief_id", "session_id", "error", "elapsed_s",
                                                  "videos", "summary")}, ensure_ascii=False) + "\n")
        s = rec["summary"] or {}
        print(f"[{rec['brief_id']}] calls={s.get('model_calls')} cost={cost} CNY time={rec['elapsed_s']}s "
              f"repair_rounds={rec['repair_rounds']} unresolved={len(rec['unresolved_issues'] or {})} "
              f"video={'yes' if rec['videos'] else 'no'} error={rec['error']}")
    print(f"total spent ~{spent:.2f} CNY; ledger: {ledger}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--briefs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--media-root", default=str(Path("~/Movies/eval_proxy").expanduser()))
    ap.add_argument("--index", default=str(ROOT / "runs" / "footage_idx"))
    ap.add_argument("--transcript", default=str(ROOT / "runs" / "speech" / "transcript.json"))
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "chinese-clip-vit-base-patch16"))
    ap.add_argument("--channels", nargs="+", default=["visual", "speech"], choices=["visual", "speech"])
    ap.add_argument("--music", default="resource/bgms/music_0057.mp3")
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--prices", default=str(ROOT / "benchmarks" / "prices.json"))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--budget-cny", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.ledger = args.ledger or str(ROOT / "benchmarks" / "results" / f"ledger_{args.label}.jsonl")
    os.chdir(ROOT)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
