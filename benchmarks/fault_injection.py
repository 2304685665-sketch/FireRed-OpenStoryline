"""
Fault injection on the multi-agent director with a real model.

For each brief, the director graph runs twice with the same seeded fault stream
(probability --rate per model call: timeout, 429, 503, or a truncated non-JSON reply):

  none       faults reach the graph directly (only the graph's own handling: a failed
             selector call becomes a critic issue and may be re-selected)
  resilient  every call goes through ResilientLLM: error classification, retries with
             backoff for transient/output failures, circuit breaker, optional fallback model

Reported per config: call success rate (a logical call succeeds if it returns parsable
JSON), brief success rate (plan complete: every beat selected, no unresolved issues, no
validator errors), retries, fallbacks, model calls and cost. No rendering.

    python benchmarks/fault_injection.py --briefs benchmarks/briefs/japan_dev.jsonl --limit 10 \
        --rate 0.3 --fallback-model qwen-plus --out benchmarks/results/fault_injection_r30.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.run_briefs import load_prices, read_briefs  # noqa: E402


class CallLog:
    """Outermost wrapper: did each logical call give the graph usable JSON?"""

    def __init__(self, inner, parse):
        self.inner, self.parse, self.ok, self.failed = inner, parse, 0, 0

    async def ainvoke(self, prompt):
        try:
            reply = await self.inner.ainvoke(prompt)
            self.parse(getattr(reply, "content", "") or "")
            self.ok += 1
            return reply
        except Exception:
            self.failed += 1
            raise


def seed_for(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


async def run_config(brief, config, args, cfg, index, encoder, lines, prices) -> dict:
    from langchain_openai import ChatOpenAI

    from benchmarks.run_director import make_llm
    from open_storyline.director.catalog import Catalog
    from open_storyline.director.graph import DirectorConfig, direct, parse_json
    from open_storyline.reliability.llm import FaultInjectingLLM, ResilientLLM
    from open_storyline.telemetry.meter import TraceWriter, instrument, load_trace, summarize
    from open_storyline.timeline.plan import EditPlan, validate

    trace = ROOT / "runs" / "fault" / f"{args.label}_{brief['id']}_{config}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.unlink(missing_ok=True)
    writer = TraceWriter(trace)
    fallback = None
    if config == "resilient" and args.fallback_model:
        fallback = instrument(ChatOpenAI(model=args.fallback_model, base_url=cfg.vlm.base_url.rstrip("/"),
                                         api_key=cfg.vlm.api_key, timeout=60, temperature=0.2, max_retries=0),
                              writer, "llm-fallback")

    def wrap(role, temperature):
        # SDK-level retries off for both configs: they would mask the injected faults alike
        real = make_llm(cfg, writer, "llm", temperature, max_retries=0)
        faulty = FaultInjectingLLM(real, args.rate, seed=seed_for(args.seed, brief["id"], role))
        llm = faulty if config == "none" else ResilientLLM(faulty, fallback=fallback, validate=parse_json,
                                                            seed=seed_for(args.seed, role))
        return CallLog(llm, parse_json), faulty, llm

    d_log, d_faulty, d_llm = wrap("director", 0.4)
    s_log, s_faulty, s_llm = wrap("selector", 0.2)
    sources = None if brief.get("media", "all") == "all" else list(brief["media"])
    catalog = Catalog(index, lines, sources=sources, encoder=encoder)
    complete, reason = False, ""
    try:
        out = await direct(brief["brief"], catalog, d_log, s_log, DirectorConfig())
        plan = EditPlan.from_dict(out["plan"])
        seg_len = {}
        for s in catalog.segments.values():
            seg_len[s.asset_path] = max(seg_len.get(s.asset_path, 0.0), s.end)
        errors = [i for i in validate(plan, seg_len) if i.level == "error"]
        n_beats = len(out["outline"]["beats"])
        complete = bool(plan.shots) and len(plan.shots) == n_beats and not out.get("issues") and not errors
        reason = "" if complete else f"shots {len(plan.shots)}/{n_beats}, unresolved {len(out.get('issues') or {})}, errors {len(errors)}"
    except Exception as e:
        reason = f"{type(e).__name__}: {str(e)[:120]}"
    s = summarize(load_trace(trace), prices) if trace.exists() else {}
    stats = [x.stats for x in (d_llm, s_llm) if hasattr(x, "stats")]
    return {
        "brief_id": brief["id"], "config": config, "complete": complete, "reason": reason,
        "calls_ok": d_log.ok + s_log.ok, "calls_failed": d_log.failed + s_log.failed,
        "injected": dict(d_faulty.injected + s_faulty.injected),
        "retries": sum((sum(st.retries.values()) for st in stats), 0),
        "fallbacks": sum(st.fallbacks for st in stats), "model_calls": s.get("model_calls"),
        "cost_cny": s.get("cost_cny"),
    }


async def main_async(args) -> None:
    from open_storyline.config import default_config_path, load_settings
    from open_storyline.director.catalog import load_transcript
    from open_storyline.retrieval.encoders import ChineseClipEncoder
    from open_storyline.retrieval.index import SegmentIndex

    cfg = load_settings(default_config_path())
    if not (cfg.llm.api_key and cfg.llm.base_url):
        sys.exit("LLM not configured: set OPENSTORYLINE_LLM_* in this terminal first")
    prices = load_prices(Path(args.prices))
    index = SegmentIndex.load(args.index)
    encoder = ChineseClipEncoder(args.model_dir)
    lines = load_transcript(args.transcript)
    rows = []
    for b in read_briefs(Path(args.briefs))[: args.limit or None]:
        for config in ("none", "resilient"):
            r = await run_config(b, config, args, cfg, index, encoder, lines, prices)
            rows.append(r)
            print(f"[{b['id']} {config:9}] complete={r['complete']} calls ok/failed={r['calls_ok']}/{r['calls_failed']} "
                  f"injected={sum(r['injected'].values())} retries={r['retries']} fallbacks={r['fallbacks']} "
                  f"cost={r['cost_cny']} {r['reason']}")
    summary = {}
    for config in ("none", "resilient"):
        rs = [r for r in rows if r["config"] == config]
        ok, failed = sum(r["calls_ok"] for r in rs), sum(r["calls_failed"] for r in rs)
        summary[config] = {
            "briefs": len(rs), "briefs_complete": sum(r["complete"] for r in rs),
            "call_success_rate": round(ok / max(1, ok + failed), 4), "calls_ok": ok, "calls_failed": failed,
            "injected": sum(sum(r["injected"].values()) for r in rs),
            "retries": sum(r["retries"] for r in rs), "fallbacks": sum(r["fallbacks"] for r in rs),
            "model_calls": sum(r["model_calls"] or 0 for r in rs),
            "cost_cny": round(sum(r["cost_cny"] or 0 for r in rs), 4),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rate": args.rate, "seed": args.seed, "fallback_model": args.fallback_model,
                               "llm": cfg.llm.model, "summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--briefs", required=True)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--label", default="r30")
    ap.add_argument("--fallback-model", default=None, help="e.g. qwen-plus on the DashScope endpoint")
    ap.add_argument("--index", default=str(ROOT / "runs" / "footage_idx"))
    ap.add_argument("--transcript", default=str(ROOT / "runs" / "speech" / "transcript.json"))
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "chinese-clip-vit-base-patch16"))
    ap.add_argument("--prices", default=str(ROOT / "benchmarks" / "prices.json"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "fault_injection.json"))
    args = ap.parse_args()
    os.chdir(ROOT)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
