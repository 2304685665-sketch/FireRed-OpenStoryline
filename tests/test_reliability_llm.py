import asyncio
import json

import pytest

from open_storyline.reliability.errors import CircuitBreaker, RetryPolicy
from open_storyline.reliability.llm import FaultInjectingLLM, MalformedOutputError, ResilientLLM


class R:
    def __init__(self, c):
        self.content = c


class Scripted:
    """Raises / returns the scripted items in order."""

    def __init__(self, *items):
        self.items, self.calls = list(items), 0

    async def ainvoke(self, prompt):
        self.calls += 1
        it = self.items.pop(0) if self.items else R('{"ok": 1}')
        if isinstance(it, BaseException):
            raise it
        return it


async def nosleep(_):
    pass


def run(llm, prompt="p"):
    return asyncio.run(llm.ainvoke(prompt))


def test_transient_errors_retried_then_succeed():
    inner = Scripted(TimeoutError("t"), RuntimeError("Error code: 429"), R('{"a": 1}'))
    llm = ResilientLLM(inner, sleep=nosleep, validate=json.loads)
    assert run(llm).content == '{"a": 1}' and inner.calls == 3
    assert llm.stats.retries["transient"] == 2


def test_malformed_json_regenerated_and_input_error_fails_fast():
    inner = Scripted(R("{broken"), R('{"a": 2}'))
    llm = ResilientLLM(inner, sleep=nosleep, validate=json.loads)
    assert run(llm).content == '{"a": 2}' and llm.stats.retries["output"] == 1
    bad = Scripted(ValueError("bad parameter"))
    llm2 = ResilientLLM(bad, sleep=nosleep, fallback=Scripted(R('{"fb": 1}')))
    with pytest.raises(ValueError):
        run(llm2)
    assert bad.calls == 1 and llm2.stats.fallbacks == 0


def test_fallback_after_retries_exhausted_and_breaker_opens():
    always = Scripted(*[RuntimeError("503 Service temporarily unavailable")] * 50)
    fb = Scripted(*[R('{"fb": 1}')] * 10)
    llm = ResilientLLM(always, fallback=fb, sleep=nosleep, policy=RetryPolicy.uniform(1),
                       breaker=CircuitBreaker(threshold=3, cooldown=999))
    assert run(llm).content == '{"fb": 1}'
    run(llm)  # breaker opens during the second call
    calls_before = always.calls
    assert run(llm).content == '{"fb": 1}' and always.calls == calls_before  # primary skipped
    assert llm.stats.breaker_rejections >= 1


def test_fault_injection_is_seeded_and_malformed_is_unparsable():
    inner = Scripted(*[R('{"x": "' + "a" * 30 + '"}')] * 200)
    f1, f2 = FaultInjectingLLM(inner, 0.3, seed=7), FaultInjectingLLM(inner, 0.3, seed=7)
    outcomes = []
    for f in (f1, f2):
        seq = []
        for _ in range(60):
            try:
                r = run(f)
                try:
                    json.loads(r.content)
                    seq.append("ok")
                except ValueError:
                    seq.append("malformed")
            except Exception as e:
                seq.append(type(e).__name__)
        outcomes.append(seq)
    assert outcomes[0] == outcomes[1]
    assert 0.15 < sum(x != "ok" for x in outcomes[0]) / 60 < 0.45
    assert "malformed" in outcomes[0]
    assert MalformedOutputError.__name__
