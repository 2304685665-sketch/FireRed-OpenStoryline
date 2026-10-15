"""
Resilient model calls and fault injection for them.

``ResilientLLM`` wraps any object with ``async ainvoke(prompt)``:
  * classifies every failure (TRANSIENT / CONTENT / OUTPUT / INPUT, see errors.py) and
    retries only the classes where a retry can help, with exponential backoff + jitter;
  * treats a reply that is not the expected JSON as an OUTPUT failure (regenerate);
  * stops calling a provider whose circuit breaker is open and falls back to a second
    model if one is configured; INPUT errors fail fast (retrying them only spends money).

``FaultInjectingLLM`` makes a real model fail on purpose, for experiments: with
probability ``rate`` per call it raises a timeout, a 429, a 503, or returns a truncated
non-JSON reply (the last one still calls the model, so it costs what a real malformed
reply costs). The random stream is seeded, so a run can be repeated exactly.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from open_storyline.reliability.errors import CircuitBreaker, CircuitOpenError, ErrorClass, RetryPolicy, classify


class MalformedOutputError(Exception):
    """Model reply is not usable (message contains the 'empty output' marker -> OUTPUT class)."""


class _Reply:
    def __init__(self, content: str):
        self.content = content


FAULT_KINDS = ("timeout", "rate_limit", "server", "malformed")


class FaultInjectingLLM:
    def __init__(self, inner, rate: float, seed: int = 0, kinds=FAULT_KINDS):
        self.inner, self.rate, self.kinds = inner, rate, tuple(kinds)
        self.rng = random.Random(seed)
        self.injected: Counter = Counter()

    async def ainvoke(self, prompt):
        if self.rng.random() < self.rate:
            kind = self.rng.choice(self.kinds)
            self.injected[kind] += 1
            if kind == "timeout":
                raise TimeoutError("injected: request timed out")
            if kind == "rate_limit":
                raise RuntimeError("injected: Error code: 429 - Too Many Requests")
            if kind == "server":
                raise RuntimeError("injected: Error code: 503 - Service temporarily unavailable")
            reply = await self.inner.ainvoke(prompt)
            text = getattr(reply, "content", "") or ""
            return _Reply(text[: max(1, len(text) // 3)].replace("}", ""))  # truncated, unparsable
        return await self.inner.ainvoke(prompt)


@dataclass
class ResilienceStats:
    calls: int = 0
    retries: Counter = field(default_factory=Counter)
    failures: Counter = field(default_factory=Counter)
    fallbacks: int = 0
    breaker_rejections: int = 0


class ResilientLLM:
    def __init__(self, primary, *, fallback=None, policy: Optional[RetryPolicy] = None,
                 breaker: Optional[CircuitBreaker] = None, validate: Optional[Callable[[str], Any]] = None,
                 sleep=asyncio.sleep, seed: int = 0):
        self.primary, self.fallback = primary, fallback
        self.policy = policy or RetryPolicy(base_delay=0.5, max_delay=8.0)
        self.breaker = breaker or CircuitBreaker(threshold=5, cooldown=20.0)
        self.validate, self.sleep = validate, sleep
        self.rng = random.Random(seed)
        self.stats = ResilienceStats()

    async def _attempt(self, llm, prompt):
        reply = await llm.ainvoke(prompt)
        if self.validate is not None:
            try:
                self.validate(getattr(reply, "content", "") or "")
            except Exception as e:
                raise MalformedOutputError(f"empty output or invalid JSON: {e}") from e
        return reply

    async def ainvoke(self, prompt):
        self.stats.calls += 1
        attempt, last = 0, None
        while True:
            try:
                self.breaker.before_call()
            except CircuitOpenError as e:
                self.stats.breaker_rejections += 1
                last = e
                break
            try:
                reply = await self._attempt(self.primary, prompt)
                self.breaker.record(None)
                return reply
            except Exception as e:  # noqa: BLE001 - classified below
                cls = classify(e)
                self.breaker.record(cls)
                last = e
                if cls is ErrorClass.INPUT or attempt >= self.policy.retries_for(cls):
                    self.stats.failures[cls.value] += 1
                    break
                self.stats.retries[cls.value] += 1
                await self.sleep(self.policy.backoff(cls, attempt, self.rng))
                attempt += 1
        input_error = not isinstance(last, CircuitOpenError) and classify(last) is ErrorClass.INPUT
        if self.fallback is not None and not input_error:
            self.stats.fallbacks += 1
            return await self._attempt(self.fallback, prompt)
        raise last
