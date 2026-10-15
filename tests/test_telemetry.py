import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from open_storyline.config import apply_env_overrides
from open_storyline.telemetry.meter import (TraceWriter, instrument, load_trace, make_tool_tracer,
                                            summarize)


class ToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _msg(text, tin=10, tout=5, **kw):
    return AIMessage(content=text, usage_metadata={"input_tokens": tin, "output_tokens": tout,
                                                   "total_tokens": tin + tout}, **kw)


def test_meter_records_calls_tokens_images(tmp_path):
    w = TraceWriter(tmp_path / "t.jsonl")
    m = instrument(GenericFakeChatModel(messages=iter([_msg("a", 100, 7), _msg("b", 50, 3)])), w, "vlm")
    img = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xx"}}
    asyncio.run(m.ainvoke([HumanMessage(content=[{"type": "text", "text": "describe"}, img, img])]))
    asyncio.run(m.ainvoke("hi"))
    recs = load_trace(w.path)
    assert [r["role"] for r in recs] == ["vlm", "vlm"]
    assert recs[0]["images"] == 2 and recs[0]["input_tokens"] == 100
    s = summarize(recs, {"": (1.0, 10.0)})
    assert s["model_calls"] == 2 and s["by_role"]["vlm"]["output_tokens"] == 10
    assert s["cost_cny"] == pytest.approx((150 * 1.0 + 10 * 10.0) / 1e6)


def test_unknown_price_or_missing_usage_makes_cost_unknown(tmp_path):
    w = TraceWriter(tmp_path / "t.jsonl")
    m = instrument(GenericFakeChatModel(messages=iter([AIMessage(content="no usage")])), w, "llm")
    asyncio.run(m.ainvoke("hi"))
    s = summarize(load_trace(w.path), {"": (1.0, 1.0)})
    assert s["cost_cny"] is None and s["calls_missing_usage"] == 1


def test_guard_aborts_model_call_before_it_is_sent(tmp_path):
    class Stop(RuntimeError):
        pass

    def guard():
        raise Stop("budget")

    w = TraceWriter(tmp_path / "t.jsonl", guard=guard)
    m = instrument(GenericFakeChatModel(messages=iter([_msg("never")])), w, "llm")
    with pytest.raises(Stop):
        asyncio.run(m.ainvoke("hi"))


def test_agent_tool_calls_traced_and_model_calls_attributed(tmp_path):
    from langchain.agents import create_agent

    w = TraceWriter(tmp_path / "t.jsonl")
    vlm = instrument(GenericFakeChatModel(messages=iter([_msg("caption", 300, 20)])), w, "vlm")

    @tool
    async def understand_clips(n: int) -> str:
        """Describe clips."""
        return (await vlm.ainvoke("describe")).content

    @tool
    def broken(x: int) -> str:
        """Always fails."""
        raise ValueError("bad input")

    llm = instrument(ToolModel(messages=iter([
        _msg("", tool_calls=[{"name": "understand_clips", "args": {"n": 1}, "id": "c1"}]),
        _msg("", tool_calls=[{"name": "broken", "args": {"x": 1}, "id": "c2"}]),
        _msg("done"),
    ])), w, "llm")
    agent = create_agent(model=llm, tools=[understand_clips, broken], middleware=[make_tool_tracer(w)])
    try:
        asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="edit my trip")]}))
    except Exception:
        pass  # whether the tool error propagates depends on the error middleware; the trace is what matters
    s = summarize(load_trace(w.path))
    assert s["tools_sequence"][:2] == ["understand_clips", "broken"]
    assert s["tool_failures"] >= 1
    assert s["model_calls_by_tool"]["understand_clips"] == 1  # the VLM call belongs to the tool
    assert s["by_role"]["llm"]["calls"] >= 2


def test_time_window_attribution_without_contextvar():
    recs = [
        {"kind": "tool_call", "tool": "split_shots", "t_start": 0.0, "ts": 1.0, "ok": True},
        {"kind": "tool_call", "tool": "understand_clips", "t_start": 1.0, "ts": 5.0, "ok": True},
        {"kind": "model_call", "role": "vlm", "model": "v", "tool": "", "t_start": 2.0, "ts": 2.5, "ok": True,
         "input_tokens": 1, "output_tokens": 1},
        {"kind": "model_call", "role": "llm", "model": "l", "tool": "", "t_start": 6.0, "ts": 6.5, "ok": True,
         "input_tokens": 1, "output_tokens": 1},
    ]
    s = summarize(recs)
    assert s["model_calls_by_tool"] == {"understand_clips": 1, "(agent)": 1}


def test_env_overrides_keep_keys_out_of_config():
    data = {"llm": {"api_key": "", "model": "x"}}
    apply_env_overrides(data, {"OPENSTORYLINE_LLM_API_KEY": "sk-1", "OPENSTORYLINE_VLM_MODEL": "qwen3-vl-plus",
                               "OPENSTORYLINE_LLM_MODEL": "  "})
    assert data["llm"] == {"api_key": "sk-1", "model": "x"}
    assert data["vlm"]["model"] == "qwen3-vl-plus"
