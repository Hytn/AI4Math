"""prover/repair/repair_strategies.py — 修复策略库"""
STRATEGIES = {
    "tactic_failed": "decompose_have_steps",
    "type_mismatch": "add_type_conversion",
    "unknown_identifier": "search_alternative_name",
    "syntax_error": "fix_syntax",
    "timeout": "simplify_proof",
}

def select_strategy(error_category: str) -> str:
    return STRATEGIES.get(error_category, "regenerate")
