"""
core/lean_checker.py — Lean 4 编译验证器

职责：接收 theorem statement + proof，提交给 Lean 4 编译器，返回结构化结果。
支持两种模式：
  1. Docker 模式 (生产推荐)：通过 docker exec 调用容器内的 Lean
  2. 本地模式 (开发调试)：直接调用本地 lean 可执行文件

设计要点：
  - 每次检查写入临时 .lean 文件，避免状态污染
  - 超时控制防止 Lean elaboration 卡死
  - 输出为 (AttemptStatus, list[LeanError], raw_stderr)
"""

from __future__ import annotations

import os
import re
import time
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

from core.models import AttemptStatus, LeanError, ErrorCategory

logger = logging.getLogger(__name__)


# ── Lean 文件模板 ──────────────────────────────────────────────
# mathlib 的 import 按需调整；初期可以用 Mathlib 全量 import
# 生产环境应改为按题目精确 import 以加速编译
LEAN_FILE_TEMPLATE = """
import Mathlib

{theorem_and_proof}
"""


class LeanChecker:
    """Lean 4 编译验证器"""

    def __init__(
        self,
        mode: str = "docker",                  # "docker" | "local"
        docker_image: str = "ai4math-lean",
        docker_container: str = "",            # 用持久容器可省去启动开销
        lean_project_dir: str = "/workspace/lean-project",  # 容器内的 lake 项目路径
        local_project_dir: str = "",           # 本地模式的 lake 项目路径
        timeout_seconds: int = 120,
    ):
        self.mode = mode
        self.docker_image = docker_image
        self.docker_container = docker_container
        self.lean_project_dir = lean_project_dir
        self.local_project_dir = local_project_dir
        self.timeout = timeout_seconds

        # 验证环境可用性
        self._validate_environment()

    def _validate_environment(self) -> None:
        """启动时检查 Lean 环境是否可用"""
        if self.mode == "local":
            if not shutil.which("lean"):
                logger.warning("lean not found in PATH; local mode may not work")
            if self.local_project_dir and not Path(self.local_project_dir).exists():
                logger.warning(f"local project dir not found: {self.local_project_dir}")
        elif self.mode == "docker":
            try:
                result = subprocess.run(
                    ["docker", "info"], capture_output=True, timeout=5
                )
                if result.returncode != 0:
                    logger.warning("docker daemon not reachable")
            except FileNotFoundError:
                logger.warning("docker CLI not found")

    def check(
        self,
        theorem_statement: str,
        proof: str,
        extra_imports: Optional[list[str]] = None,
    ) -> tuple[AttemptStatus, list[LeanError], str, str, int]:
        """
        验证一个 proof 是否通过 Lean 4 内核。

        Args:
            theorem_statement: Lean 4 的 theorem 声明 (不含 proof 体)
            proof:             Lean 4 proof 代码 (含 := by ... 或 := ...)
            extra_imports:     额外的 import 语句

        Returns:
            (status, errors, stdout, stderr, check_ms)
        """
        # 拼接完整 Lean 文件
        full_code = self._assemble_lean_file(theorem_statement, proof, extra_imports)

        start = time.time()

        if self.mode == "docker":
            status, stdout, stderr = self._check_docker(full_code)
        else:
            status, stdout, stderr = self._check_local(full_code)

        check_ms = int((time.time() - start) * 1000)

        # 解析错误
        if status == AttemptStatus.SUCCESS:
            errors = []
        else:
            errors = parse_lean_errors(stderr)

        return status, errors, stdout, stderr, check_ms

    def _assemble_lean_file(
        self,
        theorem_statement: str,
        proof: str,
        extra_imports: Optional[list[str]] = None,
    ) -> str:
        """拼装完整的 .lean 文件内容"""
        # 如果 proof 不以 := 开头，自动补上
        proof_stripped = proof.strip()
        if not proof_stripped.startswith(":=") and not proof_stripped.startswith("by"):
            # 假设 proof 是 tactic block
            if not proof_stripped.startswith("by"):
                proof_stripped = f"by\n{proof_stripped}"
            proof_stripped = f":= {proof_stripped}"

        theorem_and_proof = f"{theorem_statement.rstrip()} {proof_stripped}"

        content = LEAN_FILE_TEMPLATE.format(theorem_and_proof=theorem_and_proof)

        if extra_imports:
            import_block = "\n".join(extra_imports)
            content = import_block + "\n" + content

        return content

    def _check_local(self, lean_code: str) -> tuple[AttemptStatus, str, str]:
        """本地模式：在本地 lake 项目中编译"""
        project_dir = self.local_project_dir or self._find_local_project()
        if not project_dir:
            return AttemptStatus.LEAN_ERROR, "", "No Lean project directory configured"

        # 写入临时文件
        check_file = Path(project_dir) / "AI4MathCheck.lean"
        try:
            check_file.write_text(lean_code, encoding="utf-8")

            result = subprocess.run(
                ["lake", "env", "lean", str(check_file)],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode == 0 and "error" not in result.stderr.lower():
                return AttemptStatus.SUCCESS, result.stdout, result.stderr
            else:
                return AttemptStatus.LEAN_ERROR, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return AttemptStatus.TIMEOUT, "", f"Lean check timed out after {self.timeout}s"
        except Exception as e:
            return AttemptStatus.LEAN_ERROR, "", f"Unexpected error: {e}"
        finally:
            if check_file.exists():
                check_file.unlink()

    def _check_docker(self, lean_code: str) -> tuple[AttemptStatus, str, str]:
        """Docker 模式：在容器内编译"""
        # 写入本地临时文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False, encoding="utf-8"
        ) as f:
            f.write(lean_code)
            tmp_path = f.name

        container_check_file = f"{self.lean_project_dir}/AI4MathCheck.lean"

        try:
            # 如果有持久容器，用 docker exec
            if self.docker_container:
                # 复制文件进容器
                subprocess.run(
                    ["docker", "cp", tmp_path, f"{self.docker_container}:{container_check_file}"],
                    check=True, capture_output=True, timeout=10,
                )
                result = subprocess.run(
                    [
                        "docker", "exec", self.docker_container,
                        "lake", "env", "lean", container_check_file,
                    ],
                    cwd="/",
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            else:
                # 用 docker run (每次启动新容器，较慢但无状态)
                result = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "-v", f"{tmp_path}:{container_check_file}",
                        self.docker_image,
                        "lake", "env", "lean", container_check_file,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

            if result.returncode == 0 and "error" not in result.stderr.lower():
                return AttemptStatus.SUCCESS, result.stdout, result.stderr
            else:
                return AttemptStatus.LEAN_ERROR, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return AttemptStatus.TIMEOUT, "", f"Lean check timed out after {self.timeout}s"
        except Exception as e:
            return AttemptStatus.LEAN_ERROR, "", f"Docker error: {e}"
        finally:
            os.unlink(tmp_path)

    def _find_local_project(self) -> str:
        """尝试在常见位置找到 lean 项目"""
        candidates = [
            Path.cwd() / "lean-project",
            Path.home() / "lean-project",
            Path.home() / ".elan" / "toolchains",
        ]
        for p in candidates:
            if (p / "lakefile.lean").exists() or (p / "lakefile.toml").exists():
                return str(p)
        return ""


# ── 错误解析 ────────────────────────────────────────────────────

# Lean 4 错误格式示例:
# AI4MathCheck.lean:5:2: error: unknown identifier 'Nat.add_comm'
# AI4MathCheck.lean:8:4: error: type mismatch

_ERROR_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*error:\s*(?P<msg>.+)$",
    re.MULTILINE,
)

_CATEGORY_PATTERNS: list[tuple[re.Pattern, ErrorCategory]] = [
    (re.compile(r"type mismatch", re.I), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"unknown (identifier|constant|namespace)", re.I), ErrorCategory.UNKNOWN_IDENTIFIER),
    (re.compile(r"tactic .+ failed", re.I), ErrorCategory.TACTIC_FAILED),
    (re.compile(r"(unsolved goals|goals accomplished)", re.I), ErrorCategory.TACTIC_FAILED),
    (re.compile(r"expected .+ got", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"unexpected token", re.I), ErrorCategory.SYNTAX_ERROR),
    (re.compile(r"import .+ not found", re.I), ErrorCategory.IMPORT_ERROR),
    (re.compile(r"unknown package", re.I), ErrorCategory.IMPORT_ERROR),
    (re.compile(r"(failed to synthesize|elaboration)", re.I), ErrorCategory.ELABORATION_ERROR),
    (re.compile(r"(timeout|deterministic timeout|maximum recursion)", re.I), ErrorCategory.TIMEOUT),
]


def parse_lean_errors(stderr: str) -> list[LeanError]:
    """从 Lean stderr 中提取结构化错误列表"""
    errors = []
    for match in _ERROR_LINE_RE.finditer(stderr):
        msg = match.group("msg").strip()
        category = ErrorCategory.OTHER
        for pattern, cat in _CATEGORY_PATTERNS:
            if pattern.search(msg):
                category = cat
                break

        errors.append(LeanError(
            category=category,
            message=msg,
            line=int(match.group("line")),
            column=int(match.group("col")),
            raw=match.group(0),
        ))

    # 如果正则没匹配到任何 error 行但 stderr 非空，兜底创建一条
    if not errors and stderr.strip():
        category = ErrorCategory.OTHER
        for pattern, cat in _CATEGORY_PATTERNS:
            if pattern.search(stderr):
                category = cat
                break
        errors.append(LeanError(
            category=category,
            message=stderr.strip()[:500],
            raw=stderr.strip()[:500],
        ))

    return errors
