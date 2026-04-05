#!/usr/bin/env python3
"""scripts/export_mathlib_premises.py — 从本地 Mathlib 导出定理声明

用法:
    # 从 Mathlib Lean4 源码目录导出
    python scripts/export_mathlib_premises.py --mathlib-src ~/.elan/toolchains/.../lib/lean4/library/ -o data/premises/mathlib_full.jsonl

    # 从 lake 项目中导出 (需要先 lake build)
    python scripts/export_mathlib_premises.py --lake-project /path/to/lean-project -o data/premises/mathlib_full.jsonl

    # 使用 Lean4 的 `lake env lean` 提取 (最准确, 需要 Lean4 环境)
    python scripts/export_mathlib_premises.py --lean-extract --lake-project /path/to/lean-project -o data/premises/mathlib_full.jsonl

输出格式 (JSONL):
    {"name": "Nat.add_comm", "statement": "∀ (n m : ℕ), n + m = m + n", "domain": "nat", "tags": ["add","comm"]}

PremiseSelector 会自动加载 data/premises/*.jsonl 中的所有文件。
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from collections import Counter


def extract_from_lean_source(src_dir: str, output_file: str,
                              max_files: int = 0):
    """Parse .lean source files for theorem/lemma declarations.

    This is a best-effort text-based extraction. For 100% accuracy,
    use --lean-extract mode which invokes the Lean4 environment.
    """
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"Error: {src_dir} does not exist")
        sys.exit(1)

    lean_files = sorted(src_path.rglob("*.lean"))
    if max_files:
        lean_files = lean_files[:max_files]

    print(f"Scanning {len(lean_files)} .lean files in {src_dir}...")

    # Pattern to match theorem/lemma declarations
    # Handles multi-line declarations up to `:=` or `where` or `|`
    decl_pattern = re.compile(
        r'^(theorem|lemma|def)\s+'       # keyword
        r'(\S+)'                          # name
        r'(.*?)'                          # signature (may span lines)
        r'(?::=|where|\|)',               # terminator
        re.MULTILINE | re.DOTALL,
    )

    # Simpler single-line pattern for most cases
    simple_pattern = re.compile(
        r'^\s*(?:@\[.*?\]\s*)?'           # optional attributes
        r'(protected\s+|private\s+)?'     # optional visibility
        r'(theorem|lemma)\s+'             # keyword
        r'(\S+)\s*'                       # name
        r'(.*?)\s*:=',                    # type signature
        re.MULTILINE,
    )

    premises = []
    seen_names = set()
    domain_counter = Counter()

    for lean_file in lean_files:
        try:
            content = lean_file.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        # Determine domain from file path
        rel_path = str(lean_file.relative_to(src_path))
        domain = _infer_domain(rel_path)

        for match in simple_pattern.finditer(content):
            _, keyword, name, signature = match.groups()
            name = name.strip()
            signature = signature.strip()

            # Skip private/internal names
            if name.startswith('_') or '.proof_' in name:
                continue
            if name in seen_names:
                continue

            # Clean up the signature
            signature = _clean_signature(signature)
            if not signature or len(signature) < 5:
                continue

            # Skip if signature is too long (likely a complex definition)
            if len(signature) > 500:
                signature = signature[:500] + "..."

            seen_names.add(name)
            tags = _infer_tags(name, signature)

            premises.append({
                "name": name,
                "statement": signature,
                "domain": domain,
                "tags": tags,
                "source": rel_path,
            })
            domain_counter[domain] += 1

    # Write output
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        for p in premises:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nExtracted {len(premises)} premises → {output_file}")
    print(f"\nDomain breakdown:")
    for domain, count in domain_counter.most_common(20):
        print(f"  {domain:<20} {count:>5}")


def extract_via_lean(project_dir: str, output_file: str):
    """Extract premises using Lean4's environment API.

    This is the most accurate method: it uses `lake env lean` to
    query the actual Lean4 environment for all theorem declarations.
    Requires a working Lean4/Mathlib installation.
    """
    lean_script = '''
import Lean
import Mathlib

open Lean in
def main : IO Unit := do
  let env ← importModules #[{module := `Mathlib}] {} 0
  let mut count := 0
  for (name, info) in env.constants.map₁.toList do
    match info with
    | .thmInfo val =>
      let nameStr := name.toString
      -- Skip internal names
      if nameStr.startsWith "_" || nameStr.containsSubstr ".proof_" then
        continue
      let typeStr := toString (← Lean.Meta.MetaM.run' (Lean.Meta.ppExpr val.type) |>.run' {} {})
      IO.println s!"{{\\"name\\": \\"{nameStr}\\", \\"statement\\": \\"{typeStr}\\"}}"
      count := count + 1
    | _ => pure ()
  IO.eprintln s!"Exported {count} theorems"
'''
    print("Extracting via Lean4 environment (this may take several minutes)...")
    try:
        result = subprocess.run(
            ["lake", "env", "lean", "--stdin"],
            input=lean_script,
            capture_output=True, text=True,
            timeout=600,  # 10 minutes
            cwd=project_dir,
        )

        if result.returncode != 0:
            print(f"Lean4 extraction failed:\n{result.stderr[:1000]}")
            sys.exit(1)

        # Parse output
        premises = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    entry = json.loads(line)
                    entry["domain"] = _infer_domain(entry["name"])
                    entry["tags"] = _infer_tags(entry["name"], entry["statement"])
                    premises.append(entry)
                except json.JSONDecodeError:
                    continue

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            for p in premises:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"Exported {len(premises)} premises → {output_file}")
        print(f"Lean stderr: {result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        print("Error: Lean4 extraction timed out (>10 minutes)")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: `lake` not found. Install Lean4 via elan first.")
        sys.exit(1)


def _infer_domain(path_or_name: str) -> str:
    """Infer mathematical domain from file path or declaration name."""
    s = path_or_name.lower()
    if any(k in s for k in ["nat", "natural", "prime", "factorial", "fib"]):
        return "nat"
    if any(k in s for k in ["int.", "integer"]):
        return "int"
    if any(k in s for k in ["real", "complex", "nnreal"]):
        return "real"
    if any(k in s for k in ["group", "monoid", "subgroup", "quotientgroup"]):
        return "algebra"
    if any(k in s for k in ["ring", "ideal", "polynomial", "field"]):
        return "algebra"
    if any(k in s for k in ["topology", "topological", "compact", "connected", "continuous"]):
        return "topology"
    if any(k in s for k in ["analysis", "deriv", "integral", "measur", "filter", "tendsto"]):
        return "analysis"
    if any(k in s for k in ["linear", "module", "submodule", "matrix", "vector"]):
        return "linalg"
    if any(k in s for k in ["finset", "multiset", "combinat"]):
        return "finset"
    if any(k in s for k in ["list", "array", "string"]):
        return "list"
    if any(k in s for k in ["set.", "set/"]):
        return "set"
    if any(k in s for k in ["order", "lattice", "monoton"]):
        return "order"
    if any(k in s for k in ["logic", "prop", "classical"]):
        return "logic"
    if any(k in s for k in ["function", "equiv", "embedding"]):
        return "function"
    return "general"


def _infer_tags(name: str, statement: str) -> list[str]:
    """Infer search tags from name and statement."""
    tags = []
    s = (name + " " + statement).lower()
    tag_keywords = [
        "comm", "assoc", "cancel", "refl", "trans", "symm", "antisymm",
        "add", "mul", "sub", "div", "pow", "neg", "inv", "abs",
        "le", "lt", "eq", "ne", "pos", "nonneg",
        "dvd", "mod", "gcd", "prime", "coprime",
        "inj", "surj", "bij", "comp",
        "mem", "union", "inter", "compl", "subset",
        "sum", "prod", "card", "range", "filter",
        "open", "closed", "compact", "continuous",
        "deriv", "integral", "limit", "tendsto",
        "zero", "one", "empty", "succ",
    ]
    for t in tag_keywords:
        if t in s:
            tags.append(t)
    return tags[:8]  # Limit tag count


def _clean_signature(sig: str) -> str:
    """Clean up a type signature extracted from source."""
    # Remove newlines and excess whitespace
    sig = re.sub(r'\s+', ' ', sig).strip()
    # Remove trailing `:=`
    sig = re.sub(r'\s*:=\s*$', '', sig)
    # Remove `by` and everything after
    sig = re.sub(r'\s+by\s+.*$', '', sig)
    return sig


def main():
    parser = argparse.ArgumentParser(
        description="Export Mathlib theorem declarations as premise database")
    parser.add_argument("--mathlib-src", default="",
                        help="Path to Mathlib Lean4 source directory")
    parser.add_argument("--lake-project", default="",
                        help="Path to lake project with Mathlib dependency")
    parser.add_argument("--lean-extract", action="store_true",
                        help="Use Lean4 environment API (most accurate)")
    parser.add_argument("-o", "--output",
                        default="data/premises/mathlib_extracted.jsonl")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max .lean files to scan (0=all)")
    args = parser.parse_args()

    if args.lean_extract:
        project_dir = args.lake_project or "."
        extract_via_lean(project_dir, args.output)
    elif args.mathlib_src:
        extract_from_lean_source(args.mathlib_src, args.output,
                                  max_files=args.max_files)
    elif args.lake_project:
        # Find Mathlib source in lake packages
        mathlib_src = os.path.join(
            args.lake_project, ".lake", "packages", "mathlib", "Mathlib")
        if os.path.isdir(mathlib_src):
            extract_from_lean_source(mathlib_src, args.output,
                                      max_files=args.max_files)
        else:
            print(f"Mathlib source not found at {mathlib_src}")
            print("Run `lake build` first to download Mathlib.")
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python scripts/export_mathlib_premises.py "
              "--lake-project ~/.ai4math/lean-project "
              "-o data/premises/mathlib_full.jsonl")


if __name__ == "__main__":
    main()
