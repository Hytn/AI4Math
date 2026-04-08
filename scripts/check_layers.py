#!/usr/bin/env python3
"""scripts/check_layers.py — 验证模块层依赖方向

规则:
  engine/  不得导入  agent/ 或 prover/   (engine 是最底层)
  prover/  不得导入  agent/              (除 TYPE_CHECKING guard)

在 CI 中运行: python scripts/check_layers.py
退出码: 0 = 通过, 1 = 违规
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RULES = [
    # (source_dir, forbidden_imports, exception_pattern)
    ("engine", ["from agent.", "import agent.", "from prover.", "import prover."],
     r"#.*noqa|TYPE_CHECKING|Deprecated|deprecated"),
]

violations = []

for source_dir, forbidden, exception_re in RULES:
    source_path = os.path.join(ROOT, source_dir)
    for dirpath, _, filenames in os.walk(source_path):
        if "__pycache__" in dirpath:
            continue
        for fname in filenames:
            if not fname.endswith(".py") or fname.endswith(".bak"):
                continue
            fpath = os.path.join(dirpath, fname)
            relpath = os.path.relpath(fpath, ROOT)
            with open(fpath) as f:
                for lineno, line in enumerate(f, 1):
                    stripped = line.strip()
                    if any(stripped.startswith(pat) for pat in forbidden):
                        if exception_re and re.search(exception_re, line):
                            continue
                        violations.append(
                            f"  {relpath}:{lineno}: {stripped}")

if violations:
    print(f"❌ {len(violations)} layer violation(s) found:")
    for v in violations:
        print(v)
    sys.exit(1)
else:
    print("✅ No layer violations found")
    print("   engine/ does not import from agent/ or prover/")
    sys.exit(0)
