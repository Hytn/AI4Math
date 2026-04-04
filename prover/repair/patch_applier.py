"""prover/repair/patch_applier.py — 将修复补丁应用到原始证明

支持多种补丁类型: 行替换、tactic 替换、整体替换。
"""
from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class Patch:
    """A single patch to apply."""
    kind: str  # "replace_line", "replace_tactic", "insert_after", "delete_line", "full_replace"
    target_line: int = 0       # 1-indexed line number
    target_text: str = ""      # text to find/replace
    replacement: str = ""      # replacement text
    description: str = ""


def apply_patches(original: str, patches: list[Patch]) -> str:
    """Apply a list of patches to the original code.

    Patches are applied in order. Line numbers refer to the ORIGINAL code.
    """
    if not patches:
        return original

    # Sort by line number (descending) to avoid index shifting
    sorted_patches = sorted(patches, key=lambda p: -p.target_line)
    lines = original.split("\n")

    for patch in sorted_patches:
        if patch.kind == "full_replace":
            return patch.replacement

        idx = patch.target_line - 1  # convert to 0-indexed

        if patch.kind == "replace_line" and 0 <= idx < len(lines):
            lines[idx] = patch.replacement

        elif patch.kind == "replace_tactic" and patch.target_text:
            for i, line in enumerate(lines):
                if patch.target_text.strip() in line.strip():
                    indent = len(line) - len(line.lstrip())
                    lines[i] = " " * indent + patch.replacement
                    break

        elif patch.kind == "insert_after" and 0 <= idx < len(lines):
            indent = len(lines[idx]) - len(lines[idx].lstrip())
            lines.insert(idx + 1, " " * indent + patch.replacement)

        elif patch.kind == "delete_line" and 0 <= idx < len(lines):
            lines.pop(idx)

    return "\n".join(lines)


def create_line_replace_patch(line: int, new_text: str,
                                description: str = "") -> Patch:
    return Patch("replace_line", target_line=line,
                 replacement=new_text, description=description)


def create_tactic_replace_patch(old_tactic: str, new_tactic: str,
                                  description: str = "") -> Patch:
    return Patch("replace_tactic", target_text=old_tactic,
                 replacement=new_tactic, description=description)


def create_full_replace_patch(new_code: str, description: str = "") -> Patch:
    return Patch("full_replace", replacement=new_code, description=description)


def diff_proofs(original: str, repaired: str) -> list[dict]:
    """Compute a simple line-level diff between original and repaired proofs."""
    orig_lines = original.strip().split("\n")
    new_lines = repaired.strip().split("\n")
    changes = []
    max_len = max(len(orig_lines), len(new_lines))
    for i in range(max_len):
        orig = orig_lines[i] if i < len(orig_lines) else ""
        new = new_lines[i] if i < len(new_lines) else ""
        if orig != new:
            changes.append({
                "line": i + 1,
                "original": orig,
                "repaired": new,
                "kind": "modified" if orig and new else ("added" if new else "deleted"),
            })
    return changes
