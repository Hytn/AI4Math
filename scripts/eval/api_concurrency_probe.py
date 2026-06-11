#!/usr/bin/env python3
"""Probe an OpenAI-compatible chat API under concurrent load.

This is intentionally separate from run_eval.py: the eval runner processes
benchmark problems serially, while this script sends many small requests at a
fixed concurrency level and reports latency/error statistics.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass


@dataclass
class CallResult:
    ok: bool
    latency_s: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


async def one_call(client, *, model: str, prompt: str, max_tokens: int,
                   omit_temperature: bool, idx: int) -> CallResult:
    messages = [
        {
            "role": "user",
            "content": (
                f"{prompt}\n\nRequest id: {idx}. "
                "Reply with exactly: OK"
            ),
        }
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if not omit_temperature:
        kwargs["temperature"] = 0

    started = time.perf_counter()
    try:
        response = await client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - started
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        return CallResult(True, latency, tokens_in, tokens_out)
    except Exception as exc:
        latency = time.perf_counter() - started
        return CallResult(False, latency, error=type(exc).__name__ + ": " + str(exc))


async def run_level(client, *, concurrency: int, total: int, model: str,
                    prompt: str, max_tokens: int,
                    omit_temperature: bool) -> list[CallResult]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(idx: int) -> CallResult:
        async with sem:
            return await one_call(
                client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                omit_temperature=omit_temperature,
                idx=idx,
            )

    tasks = [asyncio.create_task(guarded(i)) for i in range(total)]
    return await asyncio.gather(*tasks)


def summarize(concurrency: int, total: int, elapsed_s: float,
              results: list[CallResult]) -> None:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    ok_lat = [r.latency_s for r in ok]
    total_in = sum(r.tokens_in for r in ok)
    total_out = sum(r.tokens_out for r in ok)
    print(
        f"concurrency={concurrency:>3} total={total:>4} "
        f"ok={len(ok):>4} err={len(bad):>4} "
        f"elapsed={elapsed_s:>7.2f}s "
        f"rps={len(results) / elapsed_s:>6.2f} "
        f"ok_rps={len(ok) / elapsed_s:>6.2f}"
    )
    if ok_lat:
        print(
            f"  latency_s: min={min(ok_lat):.2f} "
            f"p50={statistics.median(ok_lat):.2f} "
            f"p95={percentile(ok_lat, 0.95):.2f} "
            f"max={max(ok_lat):.2f} "
            f"tokens_in={total_in} tokens_out={total_out}"
        )
    if bad:
        counts: dict[str, int] = {}
        for r in bad:
            head = r.error.replace("\n", " ")[:240]
            counts[head] = counts.get(head, 0) + 1
        print("  errors:")
        for msg, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    {count}x {msg}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concurrent load probe for an OpenAI-compatible chat API.")
    parser.add_argument("--api-base", default="https://sjturethinklab.cn/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="anthropic/claude-opus-4-8")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument(
        "--sweep",
        default="",
        help="Comma-separated concurrency levels, e.g. 1,2,4,8,16,32.",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=0,
        help="When --sweep is set, override --requests for every level.",
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument(
        "--prompt",
        default="This is a short API concurrency probe.",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set {args.api_key_env}=... or pass --api-key.")

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai>=1.40.0") from exc

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.api_base,
        timeout=args.timeout,
        max_retries=0,
    )
    levels = (
        [int(x) for x in args.sweep.split(",") if x.strip()]
        if args.sweep else [args.concurrency]
    )

    try:
        for level in levels:
            total = args.requests_per_level or args.requests
            total = max(total, level)
            started = time.perf_counter()
            results = await run_level(
                client,
                concurrency=level,
                total=total,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                omit_temperature=args.omit_temperature,
            )
            elapsed = time.perf_counter() - started
            summarize(level, total, elapsed, results)
    finally:
        await client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
