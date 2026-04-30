# V5 Improvement Plan

Closes acknowledged V5+ gaps from `INFRA_MERGE_V3_REPORT.md` § "What's still V5+ work" and `REFACTOR_REPORT.md` § 九.

## Improvements

1. **Cross-problem dialog retrieval** (V5+ gap #2 / REFACTOR §九.4)
   - New: `knowledge/dialog_index.py` — TF-IDF index over saved dialogs
   - New: `KnowledgeReader.find_similar_dialogs()` + `render_similar_dialogs_for_prompt()`
   - New: `ObservationPolicy.inject_similar_dialogs` flag (default False, opt-in)
   - Wire through `_build_initial_message` so the LLM sees "last time on a similar theorem you used X" demonstrations
   - Tests: 12+ pinning ingest, retrieve, render, and end-to-end injection

2. **YAML profile templates** (V5+ gap #1 / REFACTOR §九.3)
   - New: `config/profiles/*.yaml` for all 12 PRESETS missing yaml templates
   - Tests: round-trip every YAML through `load_profile_from_yaml`, compare to corresponding PRESET

3. **LLM-based autoformalizer** (closes the "5-pattern heuristic is silly" gap)
   - New: `prover/unified/llm_autoformalizer.py` — wraps an LLMProvider as a real NL→FL translator for `NLExistenceBridgeTool`
   - Factory `make_llm_autoformalizer(llm)` returns the registered callable
   - Tests: with mock LLM, verifies translation flows through `register_autoformalizer` correctly

4. **Legacy file cleanup** (V5+ gap #3 / REFACTOR §九.5)
   - Delete unreferenced `prover/pipeline/heterogeneous_engine_legacy.py` (verified zero refs)
   - Keep `proof_loop_legacy.py` (still referenced as fallback)

5. **Test resilience for missing data/**
   - The 3 tests in `test_engine_regression.py::TestExpandedPremises` should `pytest.skip` cleanly when `data/premises/` is absent

## Constraints

- Backwards compatible: every change adds new capability, never removes
- Fail-soft: every new code path tolerates missing dependencies
- Test gates: zero regressions on the existing 1088-pass baseline
- Single-file dialog.json contract: never violated
