"""prover/models.py — 证明相关核心数据模型"""
from __future__ import annotations
import time
import json
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    LEAN_ERROR = "lean_error"
    TIMEOUT = "timeout"
    LLM_ERROR = "llm_error"


class ErrorCategory(str, Enum):
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
    category: ErrorCategory
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    raw: str = ""
    expected_type: str = ""
    actual_type: str = ""
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ProofAttempt:
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    attempt_number: int = 0
    generated_proof: str = ""
    prompt_summary: str = ""
    llm_model: str = ""
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_latency_ms: int = 0
    lean_result: AttemptStatus = AttemptStatus.LEAN_ERROR
    lean_errors: list[LeanError] = field(default_factory=list)
    lean_stderr: str = ""
    lean_check_ms: int = 0
    retrieved_premises: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    repair_rounds: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lean_result"] = self.lean_result.value
        d["lean_errors"] = [
            {**asdict(e), "category": e.category.value}
            for e in self.lean_errors
        ]
        return d


@dataclass
class ProofTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    problem_id: str = ""
    problem_name: str = ""
    theorem_statement: str = ""
    natural_language: str = ""
    attempts: list[ProofAttempt] = field(default_factory=list)
    solved: bool = False
    total_attempts: int = 0
    total_tokens: int = 0
    total_duration_ms: int = 0
    successful_proof: str = ""
    strategy_path: list[str] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    error_distribution: dict[str, int] = field(default_factory=dict)

    def add_attempt(self, a: ProofAttempt):
        a.finished_at = time.time()
        self.attempts.append(a)
        self.total_attempts = len(self.attempts)
        self.total_tokens += a.llm_tokens_in + a.llm_tokens_out
        for e in a.lean_errors:
            cat = e.category.value
            self.error_distribution[cat] = self.error_distribution.get(cat, 0) + 1
        if a.lean_result == AttemptStatus.SUCCESS:
            self.solved = True
            self.successful_proof = a.generated_proof

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "problem_id": self.problem_id,
            "problem_name": self.problem_name,
            "theorem_statement": self.theorem_statement,
            "solved": self.solved,
            "total_attempts": self.total_attempts,
            "total_tokens": self.total_tokens,
            "strategy_path": self.strategy_path,
            "successful_proof": self.successful_proof,
            "error_distribution": self.error_distribution,
            "attempts": [
                a.to_dict() if hasattr(a, 'to_dict') else a
                for a in self.attempts
            ],
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class BenchmarkProblem:
    problem_id: str
    name: str
    theorem_statement: str
    difficulty: str = "unknown"
    source: str = ""
    natural_language: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    benchmark: str
    split: str
    total_problems: int
    solved: int
    solve_rate: float
    total_tokens: int
    total_duration_ms: int
    avg_attempts: float
    per_problem: list[dict] = field(default_factory=list)
    error_distribution: dict[str, int] = field(default_factory=dict)
    per_difficulty: dict[str, dict] = field(default_factory=dict)
    avg_repair_rounds: float = 0.0
    median_solve_tokens: int = 0

    def summary(self) -> str:
        return (
            f"[{self.benchmark}/{self.split}] "
            f"Solved {self.solved}/{self.total_problems} "
            f"({self.solve_rate:.1%}) | "
            f"Tokens: {self.total_tokens:,} | "
            f"Avg attempts: {self.avg_attempts:.1f}"
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
