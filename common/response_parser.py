"""common/response_parser.py — LLM response extraction utilities (shared)"""
from __future__ import annotations
import re, json

def extract_lean_code(response: str) -> str:
    for pattern in [r"```lean\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()
    lines = response.strip().split("\n")
    return "\n".join(l for l in lines if not l.startswith("**") and not l.startswith("##")).strip()

def extract_json(response: str) -> dict:
    try:
        m = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL)
        return json.loads(m.group(1) if m else response)
    except (json.JSONDecodeError, AttributeError):
        return {}

def extract_sorry_blocks(lean_code: str) -> list[dict]:
    blocks = []
    for i, line in enumerate(lean_code.split("\n")):
        if "sorry" in line:
            blocks.append({"line": i + 1, "content": line.strip()})
    return blocks
