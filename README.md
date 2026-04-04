# AI4Math — Formal Proof Agent Platform

> A Claude-driven agent that constructs, verifies, and evolves Lean 4 formal proofs.

## Architecture

```
ai4math/
├── agent/                  # Universal agent infrastructure
│   ├── brain/              #   Claude LLM + multi-role prompting
│   ├── memory/             #   Working + episodic memory
│   ├── tools/              #   Tool registry + Lean automation + CAS bridge
│   ├── executor/           #   Lean Docker sandbox + resource limits
│   ├── context/            #   Token window management + compression
│   └── strategy/           #   Meta-controller + Light/Medium/Heavy modes
│
├── prover/                 # Theorem proving pipeline
│   ├── pipeline/           #   Orchestrator + rollout/sequential engines
│   ├── sketch/             #   Proof plan + hypothesis generation
│   ├── premise/            #   Premise retrieval (BM25/embedding/hybrid)
│   ├── codegen/            #   Sorry scaffold → tactic gen → code formatting
│   ├── verifier/           #   Lean checker + REPL + error parser + integrity
│   ├── repair/             #   Error diagnosis → fix generation → patching
│   ├── lemma_bank/         #   Cross-rollout lemma extraction + sharing
│   ├── conjecture/         #   Active conjecture generation + verification
│   ├── decompose/          #   Theorem → sub-goals → compose proof
│   └── formalize/          #   NL → Lean 4 auto-formalization
│
├── benchmarks/             # 7 datasets + evaluation
│   ├── datasets/           #   miniF2F, PutnamBench, FATE-M/H/X, ProofNet, FormalMATH
│   ├── eval_runner.py      #   Batch evaluation executor
│   └── metrics.py          #   pass@k, solve rate, token efficiency
│
├── knowledge/              # Mathlib RAG knowledge base
├── config/                 # YAML configs + experiment presets
├── eval.sh                 # One-command evaluation for any benchmark
└── web/docs/               # Investor demo (GitHub Pages)
```

## Quick Start

```bash
# Smoke test (no Lean / no API key needed)
python scripts/smoke_test.py

# Single problem (mock mode)
python run_single.py --builtin nat_add_comm --provider mock

# Batch evaluation
./eval.sh builtin
./eval.sh putnambench --limit 10
./eval.sh all
```

## Demo

Deploy to GitHub Pages: Settings → Pages → Source: `main`, folder: `/web/docs`

## Key Design Decisions

1. **agent/ vs prover/ separation**: Universal agent capabilities decoupled from theorem-proving logic
2. **Sorry-based scaffolding**: Generate proof skeleton first, then close each sorry incrementally
3. **Interactive REPL**: Tactic-by-tactic Lean interaction, not just whole-file compilation
4. **Strategy escalation**: Light (1hr) → Medium (hours) → Heavy (days) with automatic upgrade
5. **Lemma banking**: Failed proofs still produce reusable sub-lemmas for future attempts
6. **152 source files, 17,600+ lines**: Each file = one responsibility. Directory structure = architecture diagram.
7. **99/99 verification tests**: All components tested — engine, search, agent, prover, pipeline.
8. **347 unit tests**: Full regression suite with de Bruijn property tests.
