"""
Per-run accounting of model and tool calls.

Every LLM/VLM request in OpenStoryline goes through a ``ChatOpenAI`` object owned
by the client process (the agent loop directly, MCP nodes via the sampling
callback). Attaching :class:`CallMeter` to those objects therefore records every
model call of a run: role (llm/vlm), model, latency, input/output tokens, and the
error class if it failed. Tool calls are recorded by :func:`trace_tool_calls`,
an agent middleware.

Records are appended as JSON lines to ``<outputs>/<session>/trace/events.jsonl``
so a crashed run still leaves its partial trace. ``summarize`` turns a trace
into the per-run numbers used for the baseline (calls, tokens, cost, time).
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler

# Name of the MCP tool currently executing in this task (set by trace_tool_calls).
CURRENT_TOOL: contextvars.ContextVar[str] = contextvars.ContextVar("storyline_current_tool", default="")


class TraceWriter:
    """Append-only JSONL sink; one per run."""

    def __init__(self, path: str | Path, guard=None):
        """``guard()`` runs before every model call; raising from it aborts the call (budget brake)."""
        self.path = Path(path)
        self.guard = guard
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _usage(response) -> tuple[int | None, int | None]:
    """Token usage from an LLMResult (works for streaming with stream_usage=True)."""
    for gens in response.generations or []:
        for g in gens:
            msg = getattr(g, "message", None)
            um = getattr(msg, "usage_metadata", None) if msg is not None else None
            if um:
                return um.get("input_tokens"), um.get("output_tokens")
    tu = (response.llm_output or {}).get("token_usage") or {}
    if tu:
        return tu.get("prompt_tokens"), tu.get("completion_tokens")
    return None, None


class CallMeter(AsyncCallbackHandler):
    """LangChain callback that writes one record per model call."""

    raise_error = True  # only the guard raises on purpose; bookkeeping below never does

    def __init__(self, writer: TraceWriter, role: str, model: str, clock=time.monotonic):
        self.writer, self.role, self.model, self.clock = writer, role, model, clock
        self._start: dict[uuid.UUID, tuple[float, str, int, float]] = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        if self.writer.guard is not None:
            self.writer.guard()
        n_images = sum(
            1
            for batch in messages
            for m in batch
            if isinstance(m.content, list)
            for part in m.content
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        self._start[run_id] = (self.clock(), CURRENT_TOOL.get(), n_images, time.time())

    async def on_llm_end(self, response, *, run_id, **kwargs):
        self._finish(run_id, response=response)

    async def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, error=error)

    def _finish(self, run_id, response=None, error=None):
        try:
            self._record(run_id, response, error)
        except Exception:  # accounting must never break a model call
            pass

    def _record(self, run_id, response, error):
        t0, tool, n_images, t_start = self._start.pop(run_id, (self.clock(), CURRENT_TOOL.get(), 0, time.time()))
        tin, tout = _usage(response) if response is not None else (None, None)
        self.writer.write({
            "kind": "model_call", "role": self.role, "model": self.model, "tool": tool,
            "images": n_images, "input_tokens": tin, "output_tokens": tout,
            "latency_s": round(self.clock() - t0, 3), "ok": error is None,
            "error": type(error).__name__ if error is not None else None, "t_start": t_start, "ts": time.time(),
        })


def instrument(llm, writer: TraceWriter | None, role: str):
    """Attach a CallMeter to a ChatOpenAI instance (no-op without a writer)."""
    if writer is None:
        return llm
    meter = CallMeter(writer, role, getattr(llm, "model_name", "") or "")
    llm.callbacks = list(llm.callbacks or []) + [meter]
    return llm


def make_tool_tracer(writer: TraceWriter | None, clock=time.monotonic):
    """Agent middleware recording every tool call (name, duration, outcome)."""
    from langchain.agents.middleware import wrap_tool_call

    @wrap_tool_call
    async def trace_tool_calls(request, handler):
        name = (request.tool_call or {}).get("name", "")
        token = CURRENT_TOOL.set(name)
        t0, t_start = clock(), time.time()
        ok, err = True, None
        try:
            result = await handler(request)
            status = getattr(result, "status", None)
            if status == "error":
                ok, err = False, "ToolMessageError"
            return result
        except BaseException as e:  # record, then propagate unchanged
            ok, err = False, type(e).__name__
            raise
        finally:
            CURRENT_TOOL.reset(token)
            if writer is not None:
                writer.write({"kind": "tool_call", "tool": name, "latency_s": round(clock() - t0, 3),
                              "ok": ok, "error": err, "t_start": t_start, "ts": time.time()})

    return trace_tool_calls


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def summarize(records: list[dict[str, Any]], prices: dict[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    """
    Per-run totals. ``prices`` maps model -> (CNY per 1M input tokens, CNY per 1M output tokens);
    cost is None when a model has no price or a call reported no usage.
    """
    prices = prices or {}
    calls = [r for r in records if r.get("kind") == "model_call"]
    tools = [r for r in records if r.get("kind") == "tool_call"]
    by_role: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "failed": 0, "input_tokens": 0,
                                                               "output_tokens": 0, "images": 0, "latency_s": 0.0})
    by_tool: dict[str, int] = defaultdict(int)
    cost, cost_known, missing_usage = 0.0, True, 0

    def owner(c):
        # MCP sampling runs outside the tool's task, so the contextvar is often empty:
        # fall back to the tool call whose time window contains the model call.
        if c.get("tool"):
            return c["tool"]
        t = c.get("t_start")
        for tc in tools:
            if t is not None and tc.get("t_start", 1e18) <= t <= tc["ts"]:
                return tc["tool"]
        return "(agent)"

    for c in calls:
        r = by_role[c["role"]]
        r["calls"] += 1
        r["failed"] += 0 if c["ok"] else 1
        r["images"] += c.get("images") or 0
        r["latency_s"] = round(r["latency_s"] + (c.get("latency_s") or 0.0), 3)
        tin, tout = c.get("input_tokens"), c.get("output_tokens")
        if tin is None or tout is None:
            missing_usage += int(c["ok"])
        r["input_tokens"] += tin or 0
        r["output_tokens"] += tout or 0
        by_tool[owner(c)] += 1
        p = prices.get(c.get("model", ""))
        if p is None or (c["ok"] and (tin is None or tout is None)):
            cost_known = False
        else:
            cost += (tin or 0) / 1e6 * p[0] + (tout or 0) / 1e6 * p[1]
    ts = [r["ts"] for r in records if "ts" in r]
    return {
        "model_calls": len(calls),
        "by_role": dict(by_role),
        "model_calls_by_tool": dict(sorted(by_tool.items(), key=lambda kv: -kv[1])),
        "tool_calls": len(tools),
        "tool_failures": sum(1 for t in tools if not t["ok"]),
        "tools_sequence": [t["tool"] for t in tools],
        "calls_missing_usage": missing_usage,
        "cost_cny": round(cost, 6) if cost_known else None,
        "wall_s": round(max(ts) - min(ts), 1) if len(ts) > 1 else 0.0,
    }
