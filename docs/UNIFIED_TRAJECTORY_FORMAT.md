# Unified Trajectory Storage — One File, Everything Inside

Every agent trajectory in AI4Math lands as a **single self-contained
`dialog.json` file**. No companion `result.json`, no separate
`meta_config.json`. Open one file → see everything an LLM agent saw,
thought, did, and ended up with.

## On-disk layout

```
results/traces/<problem_id>/
└── dialog.json     ← {schema_version, meta, messages, result}
```

That's it. One file per task.

## Schema (v2.0)

```jsonc
{
  "schema_version": "2.0",

  "meta": {
    // Task identity
    "task_id":            "...",
    "problem_id":         "...",
    "problem_name":       "...",
    "theorem_statement":  "theorem t (n : ℕ) : n + 0 = n",
    "informal_statement": "...",

    // Execution
    "model":              "qwen3-32b",
    "provider":           "local",
    "started_at":         "2026-04-29T08:35:28Z",
    "finished_at":        "2026-04-29T08:35:30Z",

    // What the agent was given
    "system_prompt":      "You are a Lean 4 theorem prover.",
    "tools": [
      {"name": "premise_search", "description": "...",
       "parameters": {...}, "server_id": "mathlib"},
      {"name": "lean_verify", "description": "...",
       "parameters": {...}, "server_id": "lean"}
    ],

    // Free-form
    "extra": {"trace_id": "...", "config_snapshot": {...}}
  },

  "messages": [
    {"role": "user", "content": "Prove: theorem t (n : ℕ) : n + 0 = n"},

    {"role": "assistant",
     "thought": "Try ring first.",
     "content": "```lean\n... := by ring\n```",
     "tool_calls": [
       {"id": "call_a1b2",
        "function": {"name": "lean_verify",
                     "arguments": "{\"code\": \"...\"}"},
        "server_id": "lean"}
     ]},

    {"role": "tool",
     "tool_call_id": "call_a1b2", "name": "lean_verify",
     "content": "{\"verified\": false, \"errors\": [...]}",
     "server_id": "lean"}
  ],

  "result": {
    "success":            true,
    "total_attempts":     2,
    "total_tokens":       332,
    "total_duration_ms":  1500,
    "successful_proof":   "theorem t (n : ℕ) : n + 0 = n := by simp",
    "termination":        "success",
    "error_distribution": {"tactic_failed": 1},
    "extra":              {"correct_count": 1, "strategy_path": [...]}
  }
}
```

The `messages` list inside is byte-for-byte AgentCPM-compatible — any
existing AgentCPM SFT pipeline only needs to read `dialog["messages"]`
and it just works.

## Saving — every trajectory class has `save_unified()`

| Class | Module |
| --- | --- |
| `ProofTrace` | `prover/models.py` |
| `Trajectory` | `sampler/trajectory.py` |
| `LoopResult` | `agent/runtime/agent_loop.py` |
| `SessionData` | `agent/persistence/session_store.py` (sidecar) |
| `ProofSessionSnapshot` | `engine/lane/proof_session_store.py` |

```python
from prover.models import ProofTrace
trace = ProofTrace(...)
trace.add_attempt(...)
trace.save_unified(
    "results/traces/my_problem",
    model="qwen3-32b",
    provider="local",
    system_prompt="You are a Lean 4 theorem prover.",
    tools=[{"name": "lean_verify", ...}],
)
# → results/traces/my_problem/dialog.json    (one file, that's all)
```

## Loading

```python
from agent.persistence import load_task, messages_of, meta_of, result_of

d = load_task("results/traces/my_problem")

# Three accessors, each works on the wrapped form OR a legacy plain list:
messages_of(d)        # list of role-tagged turns
meta_of(d)            # {"system_prompt": ..., "tools": [...], "model": ...}
result_of(d)          # {"success": true, "total_tokens": ..., ...}
```

## SFT export — system prompt comes from the file itself

Because `meta.system_prompt` lives in the dialog, no external argument
is needed:

```python
from agent.persistence import collect_dialogs, dialogs_to_sft_jsonl

items = collect_dialogs("results/traces")
dialogs_to_sft_jsonl(
    [d for _, d in items],
    "data/sft.jsonl",
    preset="qwen3",     # or "agentcpm" / "openai"
)
```

The `qwen3` preset emits `<think>...</think>` for `thought`,
`<tool_call>...</tool_call>` for invocations,
`<tool_response>...</tool_response>` for results, and rewraps
` ```lean ... ``` ` blocks as `<code>...</code>` so the runtime can
auto-extract Lean code at inference time.

The `agentcpm` preset matches AgentCPM's training format exactly. The
`openai` preset emits a messages-array JSONL suitable for OpenAI
fine-tuning.

Per-segment `trainable: bool` flags ride alongside the rendered text
so the tokenizer can build the standard `labels` mask (-100 vs real
ID) without re-parsing.

## Backward compatibility

Old code paths still work:
- `trace.save("path/<id>.json")` — legacy single-file dump (unchanged).
- `messages_of(...)`, `meta_of(...)`, `result_of(...)` — accept either
  the wrapped form or a legacy plain message list.
- `save_dialog(plain_list, path)` — a legacy plain list is auto-wrapped
  on save.
- `load_dialog(path)` — auto-upgrades a schema-1.0 file (raw list) into
  the wrapped form on load.
