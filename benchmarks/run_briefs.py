"""
Headless batch runner: execute editing briefs through the agent and record a trace per run.

Each brief runs in a fresh session with its own media folder (symlinks to the
chosen source files), so runs are independent and repeatable. Every model call
and tool call is written to ``<outputs>/<session>/trace/events.jsonl``; a
``run.json`` next to it stores the brief, model configuration, git commit,
turns, final reply, produced videos and the summarized numbers.

If a turn ends without a rendered video (the agent asked for confirmation or only
presented a plan), the runner answers with one fixed reply (``AUTO_REPLY``) and
continues, at most ``--max-turns`` times. The same
policy is used for every system being compared.

Budget brake: before each brief the cost of finished runs is summed; the batch
stops once it reaches ``--budget-cny``. Inside a run, the trace writer's guard refuses
new model calls (agent and VLM alike) once the running total crosses the budget.

    export OPENSTORYLINE_LLM_API_KEY=...  OPENSTORYLINE_VLM_API_KEY=...
    python -m open_storyline.mcp.server &          # MCP server, as in run.sh
    python benchmarks/run_briefs.py --briefs benchmarks/briefs/japan_dev.jsonl \
        --media-root ~/Movies/eval_proxy --label baseline --budget-cny 20 --limit 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # upstream modules also import themselves as `src.open_storyline...`

AUTO_REPLY = "请按你的判断直接完成剪辑并导出成片，不需要再向我确认。"


class BudgetExceeded(RuntimeError):
    pass


def load_prices(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {m: (v["input_per_m"], v["output_per_m"]) for m, v in raw["models"].items()}


def read_briefs(path: Path) -> list[dict]:
    briefs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [b["id"] for b in briefs]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate brief ids")
    return briefs


def prepare_media(brief: dict, media_root: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in media_root.iterdir() if p.is_file() and not p.name.startswith(".")) \
        if brief.get("media", "all") == "all" else list(brief["media"])
    for n in names:
        src = media_root / n
        if not src.exists():
            raise FileNotFoundError(f"brief {brief['id']}: {src} not found")
        link = dest / n
        if not link.exists():
            link.symlink_to(src.resolve())
    return names


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def produced_videos(run_dir: Path) -> list[str]:
    return sorted(str(p) for p in run_dir.rglob("*.mp4"))


async def run_one(brief: dict, args, prices, spent_before: float) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from open_storyline.agent import ClientContext, build_agent
    from open_storyline.config import default_config_path, load_settings
    from open_storyline.mcp.hooks.node_interceptors import ToolInterceptor
    from open_storyline.storage.agent_memory import ArtifactStore
    from open_storyline.telemetry.meter import TraceWriter, load_trace, summarize
    from open_storyline.utils.media_handler import scan_media_dir
    from open_storyline.utils.prompts import get_prompt

    cfg = load_settings(default_config_path())
    session_id = f"{args.label}_{brief['id']}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    outputs = Path(cfg.project.outputs_dir)
    run_dir = outputs / session_id
    trace_path = run_dir / "trace" / "events.jsonl"
    def budget_check() -> None:
        s = summarize(load_trace(trace_path), prices) if trace_path.exists() else {"cost_cny": 0.0}
        if (s["cost_cny"] or 0.0) + spent_before >= args.budget_cny:
            raise BudgetExceeded(f"budget {args.budget_cny} CNY reached")

    writer = TraceWriter(trace_path, guard=budget_check)

    media_dir = Path(cfg.project.media_dir) / "briefs" / session_id
    media = prepare_media(brief, Path(args.media_root).expanduser(), media_dir)

    store = ArtifactStore(cfg.project.outputs_dir, session_id=session_id)
    agent, node_manager = await build_agent(
        cfg=cfg, session_id=session_id, store=store, trace=writer,
        tool_interceptors=[ToolInterceptor.inject_media_content_before,
                           ToolInterceptor.save_media_content_after, ToolInterceptor.inject_tts_config],
    )
    context = ClientContext(cfg=cfg, session_id=session_id, media_dir=str(media_dir), bgm_dir=cfg.project.bgm_dir,
                            outputs_dir=cfg.project.outputs_dir, node_manager=node_manager,
                            chat_model_key=cfg.llm.model, vlm_model_key=cfg.vlm.model, lang=args.lang)
    messages = [
        SystemMessage(content=get_prompt("instruction.system", lang=args.lang)),
        SystemMessage(content=f"【User media statistics】{json.dumps(scan_media_dir(media_dir), ensure_ascii=False)}"),
        HumanMessage(content=brief["brief"]),
    ]
    turns, final_text, error = 0, "", None
    t0 = time.time()
    try:
        while True:
            turns += 1
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": messages}, context=context),
                timeout=args.timeout_s,
            )
            messages = result["messages"]
            final_text = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "") or ""
            # Continue until a video exists: the agent often ends a turn by presenting its plan
            # or asking for confirmation without a question mark, so text heuristics are unreliable.
            if produced_videos(run_dir) or turns > args.max_turns:
                break
            messages.append(HumanMessage(content=AUTO_REPLY))
    except BudgetExceeded as e:
        error = f"BudgetExceeded: {e}"
    except asyncio.TimeoutError:
        error = f"Timeout after {args.timeout_s}s"
    except Exception as e:  # keep the batch going; the trace shows where it failed
        error = f"{type(e).__name__}: {e}"

    videos = produced_videos(run_dir)
    summary = summarize(load_trace(trace_path), prices) if trace_path.exists() else {}
    record = {
        "label": args.label, "brief_id": brief["id"], "brief": brief["brief"], "session_id": session_id,
        "media": media, "commit": git_commit(),
        "models": {"llm": cfg.llm.model, "llm_temperature": cfg.llm.temperature,
                   "vlm": cfg.vlm.model, "vlm_temperature": cfg.vlm.temperature},
        "turns": turns, "auto_replies": max(0, turns - 1), "error": error,
        "final_text": final_text[-2000:], "videos": videos, "elapsed_s": round(time.time() - t0, 1),
        "summary": summary,
    }
    (run_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return record


async def main_async(args) -> None:
    prices = load_prices(Path(args.prices))
    briefs = read_briefs(Path(args.briefs))[: args.limit or None]
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for b in briefs:
        if spent >= args.budget_cny:
            print(f"budget reached ({spent:.2f} CNY); stopping before {b['id']}")
            break
        rec = await run_one(b, args, prices, spent)
        cost = (rec["summary"] or {}).get("cost_cny")
        spent += cost or 0.0
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: rec[k] for k in ("label", "brief_id", "session_id", "error", "elapsed_s",
                                                  "videos", "summary")}, ensure_ascii=False) + "\n")
        s = rec["summary"] or {}
        print(f"[{rec['brief_id']}] calls={s.get('model_calls')} tools={s.get('tool_calls')} "
              f"cost={cost if cost is not None else 'unknown'} CNY time={rec['elapsed_s']}s "
              f"videos={len(rec['videos'])} error={rec['error']}")
    print(f"total spent ~{spent:.2f} CNY; ledger: {ledger}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--briefs", required=True)
    ap.add_argument("--media-root", required=True)
    ap.add_argument("--label", required=True, help="system name, e.g. baseline / multiagent")
    ap.add_argument("--prices", default=str(ROOT / "benchmarks" / "prices.json"))
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--budget-cny", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--lang", default="zh")
    args = ap.parse_args()
    args.ledger = args.ledger or str(ROOT / "benchmarks" / "results" / f"ledger_{args.label}.jsonl")
    os.chdir(ROOT)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
