"""agent/context/error_summarizer.py — 错误历史压缩"""
from __future__ import annotations

def summarize_round_errors(round_num: int, results: list[dict]) -> str:
    n = len(results)
    successes = sum(1 for r in results if r.get("success"))
    cats = {}
    for r in results:
        for e in r.get("errors", []):
            c = e.get("category", "other")
            cats[c] = cats.get(c, 0) + 1
    parts = [f"Round {round_num}: {successes}/{n} succeeded."]
    if cats:
        parts.append("Error distribution:")
        for c, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            parts.append(f"  - {c}: {cnt}")
    parts.append("Try fundamentally different approaches for the next round.")
    return "\n".join(parts)

def summarize_error_history(history: list[tuple[str, list]], max_rounds: int = 3) -> str:
    if not history: return ""
    recent = history[-max_rounds:]
    parts = ["## Previous attempts (most recent last)\n"]
    for i, (proof, errors) in enumerate(recent, 1):
        preview = "\n".join(proof.strip().split("\n")[:5])
        err_str = "; ".join(f"[{e.get('category','?')}] {e.get('message','')[:80]}" for e in errors[:3])
        parts.append(f"### Attempt {i}\n```lean\n{preview}\n```\nErrors: {err_str}\n")
    return "\n".join(parts)
