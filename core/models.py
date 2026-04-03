"""
core/models.py — 核心数据模型

所有模块间流动的数据结构都在此定义。
设计原则：每个 ProofAttempt 包含完整的可审计信息，可序列化为 JSON。
"""

from __future__ import annotations

import time
import json
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    LEAN_ERROR = "lean_error"
    TIMEOUT = "timeout"
    LLM_ERROR = "llm_error"


class ErrorCategory(str, Enum):
    """Lean 编译器报错的粗粒度分类"""
    TYPE_MISMATCH = "type_mismatch"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    TACTIC_FAILED = "tactic_failed"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    IMPORT_ERROR = "import_error"
    ELABORATION_ERROR = "elaboration_error"
    OTHER = "other"


@dataclass
class LeanError:
    """Lean 编译器的一条结构化错误"""
    category: ErrorCategory
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    raw: str = ""

    def to_prompt_str(self) -> str:
        """生成适合放入 LLM prompt 的错误描述"""
        loc = f" (line {self.line})" if self.line else ""
        return f"[{self.category.value}]{loc}: {self.message}"


@dataclass
class ProofAttempt:
    """单次证明尝试的完整记录"""
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attempt_number: int = 0

    # LLM 侧
    prompt_summary: str = ""          # prompt 的摘要 (不存全量避免日志膨胀)
    full_prompt: str = ""             # 完整 prompt (可选保存)
    generated_proof: str = ""         # LLM 生成的 proof 代码
    llm_model: str = ""
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_latency_ms: int = 0

    # Lean 侧
    lean_result: AttemptStatus = AttemptStatus.LEAN_ERROR
    lean_errors: list[LeanError] = field(default_factory=list)
    lean_stdout: str = ""
    lean_stderr: str = ""
    lean_check_ms: int = 0

    # 检索侧
    retrieved_premises: list[str] = field(default_factory=list)

    # 时间戳
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    @property
    def duration_ms(self) -> int:
        if self.finished_at > 0:
            return int((self.finished_at - self.started_at) * 1000)
        return 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # enum 序列化
        d["lean_result"] = self.lean_result.value
        d["lean_errors"] = [
            {**asdict(e), "category": e.category.value}
            for e in self.lean_errors
        ]
        return d


@dataclass
class ProofTrace:
    """一道题的完整证明轨迹（含所有尝试）"""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    problem_id: str = ""
    problem_name: str = ""
    theorem_statement: str = ""       # Lean 4 形式化的 theorem header
    natural_language: str = ""        # 自然语言描述 (如有)

    attempts: list[ProofAttempt] = field(default_factory=list)

    # 最终结果
    solved: bool = False
    total_attempts: int = 0
    total_tokens: int = 0
    total_duration_ms: int = 0
    successful_proof: str = ""

    # 配置快照
    config_snapshot: dict = field(default_factory=dict)

    def add_attempt(self, attempt: ProofAttempt) -> None:
        attempt.finished_at = time.time()
        self.attempts.append(attempt)
        self.total_attempts = len(self.attempts)
        self.total_tokens += attempt.llm_tokens_in + attempt.llm_tokens_out
        self.total_duration_ms += attempt.duration_ms

        if attempt.lean_result == AttemptStatus.SUCCESS:
            self.solved = True
            self.successful_proof = attempt.generated_proof

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "problem_id": self.problem_id,
            "problem_name": self.problem_name,
            "theorem_statement": self.theorem_statement,
            "natural_language": self.natural_language,
            "solved": self.solved,
            "total_attempts": self.total_attempts,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "successful_proof": self.successful_proof,
            "config_snapshot": self.config_snapshot,
            "attempts": [a.to_dict() for a in self.attempts],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ProofTrace":
        with open(path) as f:
            data = json.load(f)
        trace = cls(
            trace_id=data["trace_id"],
            problem_id=data["problem_id"],
            problem_name=data["problem_name"],
            theorem_statement=data["theorem_statement"],
            natural_language=data.get("natural_language", ""),
            solved=data["solved"],
            total_attempts=data["total_attempts"],
            total_tokens=data["total_tokens"],
            total_duration_ms=data["total_duration_ms"],
            successful_proof=data.get("successful_proof", ""),
            config_snapshot=data.get("config_snapshot", {}),
        )
        # 不还原 attempts 的完整对象，保持 dict 形式即可用于展示
        trace.attempts = data.get("attempts", [])
        return trace


@dataclass
class BenchmarkProblem:
    """基准题目"""
    problem_id: str
    name: str
    theorem_statement: str        # 完整的 Lean 4 theorem 声明 (含 import)
    difficulty: str = "unknown"   # easy / medium / hard
    source: str = ""              # "miniF2F" / "PutnamBench" / ...
    natural_language: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """一次评测的汇总结果"""
    benchmark: str
    split: str
    total_problems: int
    solved: int
    solve_rate: float
    total_tokens: int
    total_duration_ms: int
    avg_attempts_per_problem: float
    config_snapshot: dict = field(default_factory=dict)
    per_problem: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.benchmark}/{self.split}] "
            f"Solved {self.solved}/{self.total_problems} "
            f"({self.solve_rate:.1%}) | "
            f"Tokens: {self.total_tokens:,} | "
            f"Time: {self.total_duration_ms/1000:.1f}s | "
            f"Avg attempts: {self.avg_attempts_per_problem:.1f}"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
