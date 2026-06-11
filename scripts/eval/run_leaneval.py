#!/usr/bin/env python3
"""scripts/eval/run_leaneval.py — lean-eval 官方口径 (comparator) 评分

职责边界 (再次强调范式差异):
  - 本框架的 lean_verify (编译 + 无 sorry) 对 lean-eval **只是预筛**;
  - 官方口径 = 把提交写进 ``generated/<id>/Submission.lean``, 由
    lean-eval 自带工具链 validate + comparator 判定。本脚本就是
    这条官方链路的批量驱动。

输入来源 (二选一):
  --from-traces results/<run>/traces/leaneval/
        从本框架的 dialog.json 里取 result.successful_proof
        (或最后一个 ```lean 块) 作为提交;
  --proofs-dir some_dir/
        目录下每题一个 <problem_id>.lean, 整文件即提交内容。

流程 (每题):
  1. 备份 generated/<id>/Submission.lean → .bak;
  2. 写入提交 (题面声明 + 证明);
  3. `lake exe lean-eval validate-submission --file generated/<id>/Submission.lean`
     (路径合法性校验, 命令可经 --validate-cmd 覆盖);
  4. comparator 评分: `lake exe lean-eval score --problem <id>`
     (lean-eval 的评分 CLI 仍在演进, 经 --score-cmd 覆盖;
      退出码 0 = comparator accepts = solved);
  5. 还原备份 (除非 --keep-submissions)。

输出: <out>/leaneval_results.json
  {problem_id: {"validated": bool, "accepted": bool, "detail": ...}, ...}
  以及总分 solved/total。

要求: data/LeanEval 已 clone 且其工具链可构建 (按其 README);
`lake` 在 PATH 上。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_LEAN_FENCE = re.compile(r"```lean4?\s*\n(.*?)```", re.DOTALL)


def proof_from_dialog(dialog_path: Path) -> str:
    """从 dialog.json 提取最终提交: 优先 result.successful_proof,
    退而取消息流里最后一个 lean fence。"""
    try:
        d = json.loads(dialog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    proof = (d.get("result") or {}).get("successful_proof") or ""
    if proof.strip():
        return proof
    for msg in reversed(d.get("messages") or []):
        content = msg.get("content")
        if isinstance(content, str):
            m = list(_LEAN_FENCE.finditer(content))
            if m:
                return m[-1].group(1)
    return ""


def collect_submissions(args) -> dict[str, str]:
    subs: dict[str, str] = {}
    if args.from_traces:
        root = Path(args.from_traces)
        for dlg in sorted(root.glob("*/dialog.json")):
            pid = dlg.parent.name
            pid = pid.removeprefix("leaneval_")
            proof = proof_from_dialog(dlg)
            if proof.strip():
                subs[pid] = proof
    if args.proofs_dir:
        for lf in sorted(Path(args.proofs_dir).glob("*.lean")):
            subs[lf.stem.removeprefix("leaneval_")] = \
                lf.read_text(encoding="utf-8")
    return subs


def run_cmd(cmd: list[str], cwd: Path, timeout: int) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except FileNotFoundError as e:
        return False, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leaneval-dir", default="data/LeanEval")
    ap.add_argument("--from-traces", default="",
                    help="results/<run>/traces/leaneval/ 目录")
    ap.add_argument("--proofs-dir", default="",
                    help="每题一个 <id>.lean 的目录")
    ap.add_argument("--out", default="results/leaneval")
    ap.add_argument("--validate-cmd",
                    default="lake exe lean-eval validate-submission "
                            "--file {submission}",
                    help="{submission} 会被替换为相对仓库的提交路径")
    ap.add_argument("--score-cmd",
                    default="lake exe lean-eval score --problem {id}",
                    help="{id} 替换为题号; 退出码 0 = comparator accepts。"
                         "lean-eval 评分 CLI 若有变化, 在此覆盖。")
    ap.add_argument("--timeout", type=int, default=600,
                    help="每题 comparator 评分超时 (秒)")
    ap.add_argument("--keep-submissions", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    repo = Path(args.leaneval_dir)
    if not (repo / "generated").is_dir():
        sys.exit(f"Error: {repo}/generated 不存在 — 请先按 lean-eval README "
                 f"完成 clone 与 comparator workspace 生成。")
    if not args.from_traces and not args.proofs_dir:
        sys.exit("Error: 需要 --from-traces 或 --proofs-dir 之一。")
    uses_lake = args.validate_cmd.strip().startswith("lake") \
        or args.score_cmd.strip().startswith("lake")
    if uses_lake and shutil.which("lake") is None:
        sys.exit("Error: 默认 comparator 命令需要 `lake` (不在 PATH)。"
                 "先 `source ~/.elan/env`, 或用 --validate-cmd/--score-cmd "
                 "指定非 lake 的 comparator 包装。")

    subs = collect_submissions(args)
    if not subs:
        sys.exit("Error: 没有收集到任何提交。")
    if args.limit:
        subs = dict(list(subs.items())[:args.limit])
    print(f"Collected {len(subs)} submissions; scoring against "
          f"comparator in {repo} ...")

    results: dict[str, dict] = {}
    solved = 0
    for pid, proof in subs.items():
        ws = repo / "generated" / pid
        sub_file = ws / "Submission.lean"
        if not ws.is_dir():
            results[pid] = {"validated": False, "accepted": False,
                            "detail": f"no workspace generated/{pid}"}
            continue
        backup = None
        if sub_file.exists():
            backup = sub_file.read_text(encoding="utf-8")
        try:
            sub_file.write_text(proof, encoding="utf-8")
            rel = sub_file.relative_to(repo)
            v_ok, v_out = run_cmd(
                shlex.split(args.validate_cmd.format(submission=rel)),
                cwd=repo, timeout=args.timeout)
            s_ok, s_out = (False, "skipped: validation failed")
            if v_ok:
                s_ok, s_out = run_cmd(
                    shlex.split(args.score_cmd.format(id=pid)),
                    cwd=repo, timeout=args.timeout)
            results[pid] = {"validated": v_ok, "accepted": s_ok,
                            "detail": (v_out if not v_ok else s_out)[-500:]}
            if s_ok:
                solved += 1
            print(f"  {pid}: validated={v_ok} accepted={s_ok}")
        finally:
            if not args.keep_submissions and backup is not None:
                sub_file.write_text(backup, encoding="utf-8")

    os.makedirs(args.out, exist_ok=True)
    out_path = Path(args.out) / "leaneval_results.json"
    out_path.write_text(json.dumps({
        "scoring": "comparator",   # 官方口径标记 (区别于 prefilter_only)
        "solved": solved,
        "total": len(subs),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nComparator-accepted: {solved}/{len(subs)}  → {out_path}")


if __name__ == "__main__":
    main()
