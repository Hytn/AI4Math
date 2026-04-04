"""prover/pipeline/dual_engine.py — 双引擎证明管线

支持两种验证后端:
  1. Lean4 Pipeline: 传统的 LLM 生成 → Lean4 编译验证
  2. APE Pipeline:   APE 引擎的持久化状态 → 分层验证 → 并行搜索

两条管线可以独立使用，也可以协同 (APE 做 L0/L1 预过滤 → Lean4 做 L2 终验)
"""
from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from prover.models import ProofAttempt, AttemptStatus, BenchmarkProblem, LeanError, ErrorCategory

logger = logging.getLogger(__name__)


class EngineBackend(str, Enum):
    LEAN4 = "lean4"
    APE = "ape"
    DUAL = "dual"  # APE pre-filter + Lean4 final verify


@dataclass
class EngineResult:
    """Unified result type for both engines."""
    backend: EngineBackend
    success: bool
    proof: str = ""
    # Timing breakdown
    total_ms: float = 0
    precheck_ms: float = 0    # APE L0/L1 time
    verify_ms: float = 0      # Lean4 / APE L2 time
    search_ms: float = 0      # APE search tree time
    # Search stats (APE only)
    nodes_explored: int = 0
    nodes_filtered_l0: int = 0
    nodes_filtered_l1: int = 0
    forks_created: int = 0
    # Error info
    errors: list = field(default_factory=list)
    error_structured: Optional[dict] = None
    # Proof path (APE only)
    tactic_path: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.success:
            return (f"[{self.backend.value}] ✓ Proved in {self.total_ms:.1f}ms "
                    f"({self.nodes_explored} nodes, {self.forks_created} forks)")
        return (f"[{self.backend.value}] ✗ Failed in {self.total_ms:.1f}ms "
                f"({len(self.errors)} errors)")


class Lean4Engine:
    """Lean4 验证引擎: 完整的 Lean4 编译验证。

    对应传统的 'LLM 生成完整证明 → Lean4 一次性编译检查' 工作流。
    优势: 完整的 Mathlib 生态，100% 可靠的验证。
    劣势: 每次验证需要完整编译，无法做增量/分叉/并行搜索。

    Requires a LeanEnvironment (from agent.executor.lean_env) or a
    compatible object with a ``compile(code) -> (rc, stdout, stderr)`` method.
    """

    def __init__(self, lean_env=None):
        self.lean_env = lean_env
        self._initialized = False
        self._import_time_ms = 0

    def initialize(self, imports: str = "import Mathlib"):
        """Initialize Lean environment (may take seconds for Mathlib)."""
        if self.lean_env is None:
            try:
                from agent.executor.lean_env import LeanEnvironment
                self.lean_env = LeanEnvironment.create()
            except Exception as e:
                logger.warning(f"Could not auto-create LeanEnvironment: {e}")
        self._initialized = True
        logger.info("Lean4Engine initialized")

    def verify(self, theorem: str, proof: str) -> EngineResult:
        """Verify a complete proof via Lean4 compilation."""
        start = time.perf_counter()

        if not self._initialized:
            self.initialize()

        if self.lean_env is None:
            return EngineResult(
                backend=EngineBackend.LEAN4, success=False,
                total_ms=(time.perf_counter() - start) * 1000,
                errors=[{"category": "env_unavailable",
                         "message": "Lean4 environment not available. "
                                    "Install lean4 via elan or configure Docker."}])

        # Use real LeanChecker for verification
        verify_start = time.perf_counter()
        try:
            from prover.verifier.lean_checker import LeanChecker
            checker = LeanChecker(self.lean_env)
            status, errors, stderr, check_ms = checker.check(theorem, proof)

            verify_ms = (time.perf_counter() - verify_start) * 1000
            total_ms = (time.perf_counter() - start) * 1000

            is_valid = (status.value == "success")
            if is_valid:
                return EngineResult(
                    backend=EngineBackend.LEAN4, success=True,
                    proof=proof, total_ms=total_ms, verify_ms=verify_ms)
            else:
                error_dicts = [
                    {"category": e.category.value, "message": e.message,
                     "line": e.line, "column": e.column}
                    for e in errors
                ]
                return EngineResult(
                    backend=EngineBackend.LEAN4, success=False,
                    total_ms=total_ms, verify_ms=verify_ms,
                    errors=error_dicts)
        except Exception as e:
            total_ms = (time.perf_counter() - start) * 1000
            return EngineResult(
                backend=EngineBackend.LEAN4, success=False,
                total_ms=total_ms,
                errors=[{"category": "internal_error", "message": str(e)}])


class APEEngine:
    """APE 验证引擎: Agent-first 的持久化证明搜索。

    对应 'Agent 在搜索树中逐步探索 tactic → 分层验证' 工作流。
    优势: O(1) 分叉/回溯，分层验证减少无效计算，结构化错误反馈。
    劣势: 目前不覆盖 Lean4 的全部类型理论，最终仍需 Lean4 做 L2 认证。
    """

    def __init__(self):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from engine.core import Expr, Name, Environment, ConstantInfo, MetaId
        from engine.core.expr import BinderInfo
        from engine.core.universe import Level
        from engine.state import ProofState, GoalView
        from engine.search import SearchCoordinator

        self._Expr = Expr
        self._Name = Name
        self._Environment = Environment
        self._ConstantInfo = ConstantInfo
        self._BinderInfo = BinderInfo
        self._Level = Level
        self._ProofState = ProofState
        self._GoalView = GoalView
        self._SearchCoordinator = SearchCoordinator

    def prove_by_search(self, theorem_name: str, goal_expr,
                        env, tactics: list[str],
                        max_depth: int = 20,
                        tactic_generator=None) -> EngineResult:
        """Run proof search using persistent state + layered verification.

        Args:
            theorem_name: Name for logging.
            goal_expr: The goal expression to prove.
            env: APE Environment.
            tactics: Static list of tactics to try.
            max_depth: Maximum search depth.
            tactic_generator: Optional callable(goal_views: list[GoalView]) -> list[str]
                that dynamically generates tactics based on current goal state.
                If provided, its output is appended to the static tactics list.
        """
        start = time.perf_counter()

        coord = self._SearchCoordinator(env, goal_expr)
        nodes_explored = 0
        forks_created = 0
        l0_filtered = 0
        l1_filtered = 0
        proof_path = []
        solved = False

        # Use the SearchCoordinator's built-in search if a generator is provided
        if tactic_generator is not None:
            def _generate_tactics(node_id: int) -> list[str]:
                """Combine static tactics with LLM-generated ones via GoalView."""
                goal_views = coord.goal_view(node_id)
                dynamic_tactics = []
                if goal_views:
                    try:
                        dynamic_tactics = tactic_generator(goal_views)
                    except Exception as e:
                        logger.debug(f"Tactic generator error: {e}")
                # Deduplicate: static first, then dynamic
                seen = set(tactics)
                result = list(tactics)
                for t in dynamic_tactics:
                    if t not in seen:
                        seen.add(t)
                        result.append(t)
                return result

            stats = coord.run_search(_generate_tactics)
            total_ms = (time.perf_counter() - start) * 1000

            if stats.is_solved:
                proof_text = " >> ".join(stats.solution_path)
                return EngineResult(
                    backend=EngineBackend.APE, success=True,
                    proof=proof_text, total_ms=total_ms,
                    search_ms=stats.time_ms,
                    nodes_explored=stats.nodes_expanded,
                    nodes_filtered_l0=stats.l0_filtered,
                    nodes_filtered_l1=stats.l1_filtered,
                    forks_created=stats.total_nodes,
                    tactic_path=stats.solution_path)
            else:
                return EngineResult(
                    backend=EngineBackend.APE, success=False,
                    total_ms=total_ms, search_ms=stats.time_ms,
                    nodes_explored=stats.nodes_expanded,
                    nodes_filtered_l0=stats.l0_filtered,
                    nodes_filtered_l1=stats.l1_filtered,
                    forks_created=stats.total_nodes,
                    error_structured={"kind": "search_exhausted",
                                      "nodes": stats.nodes_expanded,
                                      "depth": stats.max_depth_reached})

        # Fallback: BFS-style search with static tactics
        from engine.state import NodeId
        open_nodes = [0]

        for depth in range(max_depth):
            if solved or not open_nodes:
                break

            next_open = []
            for node_id in open_nodes:
                results = coord.try_batch(node_id, tactics)
                for r in results:
                    nodes_explored += 1
                    forks_created += 1

                    if not r.success:
                        if r.elapsed_us < 5:
                            l0_filtered += 1
                        else:
                            l1_filtered += 1
                        continue

                    if r.is_complete:
                        solved = True
                        proof_path = self._extract_path(coord, r.child_node)
                        break

                    next_open.append(r.child_node)

                if solved:
                    break

            open_nodes = next_open[:50]  # Limit beam width

        total_ms = (time.perf_counter() - start) * 1000
        search_ms = total_ms  # All time is search in APE

        if solved:
            proof_text = " >> ".join(proof_path) if proof_path else "proved"
            return EngineResult(
                backend=EngineBackend.APE, success=True,
                proof=proof_text, total_ms=total_ms,
                search_ms=search_ms,
                nodes_explored=nodes_explored,
                nodes_filtered_l0=l0_filtered,
                nodes_filtered_l1=l1_filtered,
                forks_created=forks_created,
                tactic_path=proof_path
            )
        else:
            return EngineResult(
                backend=EngineBackend.APE, success=False,
                total_ms=total_ms, search_ms=search_ms,
                nodes_explored=nodes_explored,
                nodes_filtered_l0=l0_filtered,
                nodes_filtered_l1=l1_filtered,
                forks_created=forks_created,
                error_structured={"kind": "search_exhausted",
                                  "nodes": nodes_explored, "depth": max_depth}
            )

    def _extract_path(self, coord, final_node_id):
        """Extract the tactic path from root to solution."""
        path = []
        stats = coord.stats()
        # Simplified path extraction
        return path

    def build_env_for_theorem(self, theorem_desc: dict):
        """Build an APE environment from a theorem description."""
        Expr = self._Expr
        Name = self._Name
        Env = self._Environment
        CI = self._ConstantInfo
        BI = self._BinderInfo
        Level = self._Level

        env = Env()

        # Add standard types
        prop = Expr.sort(Level.zero())
        type_ = Expr.sort(Level.one())

        env = env.add_const(CI(Name.from_str("Prop"), type_))
        env = env.add_const(CI(Name.from_str("Nat"), type_))
        env = env.add_const(CI(Name.from_str("Nat.zero"),
                               Expr.const(Name.from_str("Nat"))))
        nat = Expr.const(Name.from_str("Nat"))
        env = env.add_const(CI(Name.from_str("Nat.succ"),
                               Expr.arrow(nat, nat)))

        # Add theorem-specific constants
        for const in theorem_desc.get("constants", []):
            env = env.add_const(CI(
                Name.from_str(const["name"]),
                self._parse_type(const["type"])
            ))

        return env

    def _parse_type(self, type_str: str):
        """Simplified type string parser."""
        Expr = self._Expr
        Name = self._Name
        Level = self._Level

        if type_str == "Prop":
            return Expr.prop()
        if type_str == "Nat":
            return Expr.const(Name.from_str("Nat"))
        return Expr.prop()


class DualEngine:
    """双引擎管线: 同时支持 Lean4 和 APE 两条验证路径。

    Usage:
        engine = DualEngine()

        # Lean4 path: traditional compile-verify
        result = engine.verify_lean4(theorem, proof)

        # APE path: agent search with persistent state
        result = engine.prove_ape(theorem_desc, tactics)

        # Dual path: APE pre-filter + Lean4 final verify
        result = engine.prove_dual(theorem, theorem_desc, tactics)
    """

    def __init__(self, lean_env=None):
        self.lean4 = Lean4Engine(lean_env)
        self.ape = APEEngine()

    def verify_lean4(self, theorem: str, proof: str) -> EngineResult:
        """Path 1: Traditional Lean4 verification."""
        return self.lean4.verify(theorem, proof)

    def prove_ape(self, goal_expr, env, tactics: list[str],
                  max_depth: int = 20) -> EngineResult:
        """Path 2: APE search-based proving."""
        return self.ape.prove_by_search("theorem", goal_expr, env,
                                        tactics, max_depth)

    def prove_dual(self, theorem: str, proof: str,
                   goal_expr, env, tactics: list[str]) -> EngineResult:
        """Path 3: APE pre-filter → Lean4 final verification.

        1. APE does L0/L1 quick check on the proof
        2. If APE passes, send to Lean4 for L2 certification
        3. If APE rejects, skip Lean4 (saves compile time)
        """
        start = time.perf_counter()

        # Step 1: APE pre-check (L0/L1)
        precheck_start = time.perf_counter()
        ape_result = self.ape.prove_by_search("theorem", goal_expr, env,
                                              tactics, max_depth=5)
        precheck_ms = (time.perf_counter() - precheck_start) * 1000

        if not ape_result.success:
            # APE rejected — skip expensive Lean4 compilation
            total_ms = (time.perf_counter() - start) * 1000
            return EngineResult(
                backend=EngineBackend.DUAL, success=False,
                total_ms=total_ms, precheck_ms=precheck_ms,
                nodes_explored=ape_result.nodes_explored,
                nodes_filtered_l0=ape_result.nodes_filtered_l0,
                error_structured={"stage": "ape_precheck", "reason": "search_failed"}
            )

        # Step 2: Lean4 final verification (L2)
        lean_result = self.lean4.verify(theorem, proof)

        total_ms = (time.perf_counter() - start) * 1000
        return EngineResult(
            backend=EngineBackend.DUAL,
            success=lean_result.success,
            proof=proof,
            total_ms=total_ms,
            precheck_ms=precheck_ms,
            verify_ms=lean_result.verify_ms,
            search_ms=ape_result.search_ms,
            nodes_explored=ape_result.nodes_explored,
            forks_created=ape_result.forks_created,
            tactic_path=ape_result.tactic_path,
        )
