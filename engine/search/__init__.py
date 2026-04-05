"""engine/search — Proof search with MCTS/UCB and best-first strategies.

Replaces the naive BFS with:
  1. UCB1 node selection (exploration vs exploitation)
  2. Backpropagation of success/failure
  3. Priority queue for best-first expansion
  4. Virtual loss for parallel coordination
  5. Configurable search strategies (bfs, dfs, best_first, mcts)
"""
from __future__ import annotations
import math
import time
import heapq
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from engine.core import Expr, Name, MetaId
from engine.state import ProofState, SearchTree, NodeId, NodeStatus, GoalView
from engine.tactic import execute_tactic, TacticResult

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    parent_node: int
    tactic: str
    success: bool
    child_node: Optional[int] = None
    new_goals: list = field(default_factory=list)
    error: Optional[dict] = None
    elapsed_us: int = 0
    is_complete: bool = False


@dataclass
class SearchConfig:
    """Configuration for proof search."""
    strategy: str = "best_first"  # bfs, dfs, best_first, mcts
    max_nodes: int = 10000
    max_depth: int = 50
    beam_width: int = 64
    ucb_c: float = 1.414          # UCB exploration constant
    virtual_loss: float = 1.0     # virtual loss for parallel search
    timeout_ms: int = 60000       # 60 second default timeout
    prior_weight: float = 0.5     # weight for LLM prior in scoring
    # ── 评分超参数 (原为硬编码魔数, 现可通过配置调优) ──
    goal_reduction_weight: float = 2.0    # goal 减少的奖励权重
    completion_bonus: float = 10.0        # 证明完成的额外奖励
    depth_penalty: float = 0.05           # 每层深度的惩罚
    speed_bonus_weight: float = 0.1       # tactic 执行速度奖励权重
    speed_bonus_threshold_us: float = 500000.0  # 参考阈值: 500ms (REPL 模式典型上限)
    # 使用对数衰减: bonus = weight * max(0, 1 - log2(elapsed / threshold + 1))
    # 这使得 1μs → ~0.1, 50ms → ~0.04, 500ms → 0.0 的梯度在
    # 本地模式和 REPL 模式下都有区分度


@dataclass
class SearchStats:
    total_nodes: int = 0
    nodes_expanded: int = 0
    nodes_pruned: int = 0
    l0_filtered: int = 0
    l1_filtered: int = 0
    max_depth_reached: int = 0
    time_ms: float = 0
    is_solved: bool = False
    solution_depth: int = 0
    solution_path: list[str] = field(default_factory=list)


class SearchCoordinator:
    """Manages proof search with configurable strategy.

    Supports:
      - Best-first search with heuristic scoring
      - MCTS with UCB1 node selection
      - Beam search with width limit
      - Backpropagation of results
    """

    def __init__(self, env, goal_type: Expr,
                 config: SearchConfig = None,
                 lean_pool: 'LeanPool' = None):
        self._env = env
        self._config = config or SearchConfig()
        state = ProofState.new(env, goal_type)
        self._tree = SearchTree(state)
        self._stats = SearchStats()
        self._start_time = time.perf_counter()
        self._lock = __import__('threading').Lock()

        # Priority queue: (negative_score, node_id) — heapq is min-heap
        self._frontier: list[tuple[float, int]] = []
        heapq.heappush(self._frontier, (0.0, 0))

        # Node scores for UCB
        self._scores: dict[int, float] = {0: 0.0}
        self._priors: dict[int, float] = {}  # LLM prior probabilities

        # Virtual loss: tracks nodes currently being explored by other workers
        # Prevents multiple workers from exploring the same node simultaneously
        self._virtual_losses: dict[int, float] = {}

        # ── REPL 池集成 ──
        # 当 lean_pool 可用时, try_tactic 通过 Lean4 REPL 执行 tactic,
        # 获得精确的类型检查结果; 否则回退到本地简化 tactic 引擎。
        self._lean_pool = lean_pool
        # 节点 ID → REPL env_id 映射 (REPL 模式下使用)
        self._node_env_map: dict[int, int] = {}
        if lean_pool and lean_pool._sessions:
            # 根节点绑定到 REPL 池的基础 env_id
            base_env = lean_pool._sessions[0].base_env_id if lean_pool._sessions else 0
            self._node_env_map[0] = base_env

    # ── Public API ──

    def goal_view(self, node_id: int) -> list[GoalView]:
        node = self._tree.get(NodeId(node_id))
        if not node:
            return []
        return [GoalView.from_goal(g, node.state) for g in node.state.goals()]

    def try_tactic(self, node_id: int, tactic_str: str,
                   prior: float = 0.5) -> ExpansionResult:
        """Try a single tactic at a node.

        When lean_pool is available, routes through Lean4 REPL for precise
        type checking. Falls back to local tactic engine otherwise.
        """
        t0 = time.perf_counter_ns()
        node = self._tree.get(NodeId(node_id))
        if not node:
            return ExpansionResult(node_id, tactic_str, False,
                                  error={"kind": "not_found"}, elapsed_us=0)

        # ── 优先使用 REPL 池 (精确验证) ──
        if self._lean_pool and node_id in self._node_env_map:
            return self._try_tactic_via_repl(
                node_id, node, tactic_str, prior, t0)

        # ── 回退到本地 tactic 引擎 (启发式) ──
        result = execute_tactic(node.state, tactic_str)
        elapsed = int((time.perf_counter_ns() - t0) / 1000)
        self._stats.nodes_expanded += 1

        if result.success:
            return self._handle_tactic_success(
                node_id, tactic_str, result.state, prior, elapsed,
                result.state.is_complete(),
                [GoalView.from_goal(g, result.state)
                 for g in result.state.goals()])

        # Tactic failed
        if elapsed < 5:
            self._stats.l0_filtered += 1
        else:
            self._stats.l1_filtered += 1

        err = None
        if result.error:
            err = {"kind": result.error.kind, "message": result.error.message}
        return ExpansionResult(node_id, tactic_str, False,
                               error=err, elapsed_us=elapsed)

    def _try_tactic_via_repl(self, node_id: int, node, tactic_str: str,
                              prior: float, t0: int) -> ExpansionResult:
        """通过 Lean4 REPL 池执行 tactic (精确验证路径)

        关键设计: 当 REPL 验证成功时, 子节点的 state 必须反映 REPL
        返回的 goal 信息, 而不是本地简化引擎的结果。

        本地引擎仅支持 18 种 tactic, 对 norm_num/linarith/field_simp 等
        常用 tactic 必然失败。如果回退到父节点的 state, 搜索树节点将
        携带错误的 goal state, 导致后续所有 tactic 尝试基于错误的状态。

        修复: 优先使用本地引擎的 state (如果成功); 否则从 REPL 返回的
        goal 字符串构造一个轻量级的代理 state, 准确表示剩余 goal 数量
        和证明完成状态。评分函数也直接使用 REPL 的 goal 信息而非依赖
        本地 TacticResult。
        """
        env_id = self._node_env_map[node_id]
        repl_result = self._lean_pool.try_tactic(env_id, tactic_str)
        elapsed = int((time.perf_counter_ns() - t0) / 1000)
        self._stats.nodes_expanded += 1

        if repl_result.success:
            complete = repl_result.is_proof_complete
            goals = [{"target": g} for g in repl_result.remaining_goals]

            # 构造子节点 state: 优先本地引擎, 降级时使用 REPL 信息
            local_result = execute_tactic(node.state, tactic_str)
            if local_result.success:
                child_state = local_result.state
            else:
                # 本地引擎不支持此 tactic — 从 REPL 结果构造代理 state
                child_state = self._build_state_from_repl(
                    node.state, repl_result.remaining_goals, complete)

            self._tree, child_id = self._tree.expand(
                NodeId(node_id), tactic_str, child_state)
            child_int = child_id.id

            # 将新 env_id 绑定到子节点
            self._node_env_map[child_int] = repl_result.new_env_id

            self._stats.total_nodes = self._tree.size()
            depth = (self._tree.get(child_id).depth
                     if self._tree.get(child_id) else 0)
            self._stats.max_depth_reached = max(
                self._stats.max_depth_reached, depth)

            # 评分: 直接使用 REPL 的 goal 信息, 不依赖本地 TacticResult
            goals_before = node.state.num_goals() if hasattr(node.state, 'num_goals') else 1
            score = self._compute_score_from_repl(
                goals_before=goals_before,
                goals_after=len(repl_result.remaining_goals),
                prior=prior,
                depth=depth,
                elapsed_us=elapsed,
            )
            self._scores[child_int] = score
            self._priors[child_int] = prior

            if complete:
                self._stats.is_solved = True
                self._stats.solution_depth = depth
                self._stats.solution_path = self._extract_path(child_int)
                self._backpropagate(child_int, success=True)
            else:
                heapq.heappush(self._frontier, (-score, child_int))
                self._backpropagate(child_int, success=False)

            return ExpansionResult(node_id, tactic_str, True, child_int,
                                  goals, elapsed_us=elapsed,
                                  is_complete=complete)

        # REPL 报告失败
        self._stats.l1_filtered += 1
        err = {"kind": repl_result.error_category,
               "message": repl_result.error_message}
        return ExpansionResult(node_id, tactic_str, False,
                               error=err, elapsed_us=elapsed)

    def _build_state_from_repl(self, parent_state: 'ProofState',
                                remaining_goals: list[str],
                                is_complete: bool) -> 'ProofState':
        """从 REPL 返回的 goal 字符串构造代理 ProofState.

        当本地 tactic 引擎不支持某个 tactic (如 norm_num, linarith) 时,
        我们仍需为搜索树节点提供一个 state 对象, 使得:
          1. state.num_goals() 返回正确的剩余 goal 数
          2. state.is_complete() 返回正确的完成状态
          3. state.goals() 返回可用于 GoalView 的 Goal 对象

        这些代理 goal 使用 REPL 返回的 goal 字符串作为 target type
        的文本表示, 而非真正的 Expr 对象。这对搜索树的结构正确性和
        评分计算已经足够。
        """
        from pyrsistent import pvector

        mc = parent_state.meta_ctx
        new_goal_ids = []

        if is_complete:
            # 证明完成 — 空 focus
            return ProofState(parent_state.env, mc, pvector([]),
                              parent_state.id, parent_state._next_fvar)

        # 为每个 REPL 返回的 goal 创建一个 meta variable
        for goal_str in remaining_goals:
            # 将 goal 字符串包装为 Expr.const (作为 target type 的占位符)
            # 这不是精确的类型论表示, 但搜索树只需要知道 goal 的数量
            # 和文本描述 — 实际验证始终通过 REPL 进行
            goal_type = Expr.const(Name.from_str(f"_repl_goal_{goal_str[:80]}"))
            mc, gid = mc.create_meta(LocalContext(), goal_type, depth=0)
            new_goal_ids.append(gid)

        return ProofState(parent_state.env, mc, pvector(new_goal_ids),
                          parent_state.id, parent_state._next_fvar)

    def _compute_score_from_repl(self, goals_before: int, goals_after: int,
                                  prior: float, depth: int,
                                  elapsed_us: int) -> float:
        """基于 REPL 结果计算节点评分 (不依赖本地 TacticResult).

        与 _compute_score 使用相同的评分逻辑和可配置超参数,
        但直接接受 goal 计数而非 TacticResult 对象。
        """
        cfg = self._config
        score = prior * cfg.prior_weight

        if goals_before > 0:
            reduction = (goals_before - goals_after) / goals_before
            score += reduction * cfg.goal_reduction_weight

        if goals_after == 0:
            score += cfg.completion_bonus

        score -= depth * cfg.depth_penalty

        if elapsed_us > 0:
            score += self._speed_bonus(elapsed_us)

        return score

    def _handle_tactic_success(self, node_id, tactic_str, new_state,
                                prior, elapsed, complete, goals):
        """处理本地 tactic 引擎的成功结果 (提取公共逻辑)"""
        self._tree, child_id = self._tree.expand(
            NodeId(node_id), tactic_str, new_state)
        child_int = child_id.id
        self._stats.total_nodes = self._tree.size()
        depth = (self._tree.get(child_id).depth
                 if self._tree.get(child_id) else 0)
        self._stats.max_depth_reached = max(
            self._stats.max_depth_reached, depth)

        from engine.tactic import TacticResult
        dummy_result = TacticResult(state=new_state)
        score = self._compute_score(child_int, dummy_result, prior, depth)
        self._scores[child_int] = score
        self._priors[child_int] = prior

        if complete:
            self._stats.is_solved = True
            self._stats.solution_depth = depth
            self._stats.solution_path = self._extract_path(child_int)
            self._backpropagate(child_int, success=True)
        else:
            heapq.heappush(self._frontier, (-score, child_int))
            self._backpropagate(child_int, success=False)

        return ExpansionResult(node_id, tactic_str, True, child_int,
                               goals, elapsed_us=elapsed,
                               is_complete=complete)

    def try_batch(self, node_id: int, tactics: list[str],
                  priors: list[float] = None) -> list[ExpansionResult]:
        """Try multiple tactics at a node."""
        if priors is None:
            priors = [0.5] * len(tactics)
        results = [self.try_tactic(node_id, t, p)
                   for t, p in zip(tactics, priors)]
        # Release virtual loss after batch expansion completes
        self.release_virtual_loss(node_id)
        return results

    def select_node(self) -> Optional[int]:
        """Select the next node to expand based on search strategy.

        Returns node_id or None if search is exhausted.
        Applies virtual loss to prevent parallel workers from selecting
        the same node simultaneously.
        """
        cfg = self._config

        # Check termination
        if self._stats.is_solved:
            return None
        if self._stats.total_nodes >= cfg.max_nodes:
            return None
        elapsed = (time.perf_counter() - self._start_time) * 1000
        if elapsed > cfg.timeout_ms:
            return None

        strategy = cfg.strategy

        with self._lock:
            if strategy == "mcts":
                node_id = self._select_ucb()
            elif strategy == "dfs":
                node_id = self._select_dfs()
            elif strategy == "bfs":
                node_id = self._select_bfs()
            else:  # best_first
                node_id = self._select_best_first()

            # Apply virtual loss to discourage other workers from selecting
            # the same node before we finish expanding it
            if node_id is not None:
                vl = self._config.virtual_loss
                self._virtual_losses[node_id] = \
                    self._virtual_losses.get(node_id, 0) + vl

        return node_id

    def release_virtual_loss(self, node_id: int):
        """Remove virtual loss after expansion is complete.

        Called automatically by try_tactic/try_batch, but can also be
        called manually if expansion is cancelled.
        """
        with self._lock:
            if node_id in self._virtual_losses:
                self._virtual_losses[node_id] -= self._config.virtual_loss
                if self._virtual_losses[node_id] <= 0:
                    del self._virtual_losses[node_id]

    def run_search(self, tactic_generator: Callable[[int], list[str]],
                   prior_generator: Callable[[int], list[float]] = None
                   ) -> SearchStats:
        """Run complete proof search.

        Args:
            tactic_generator: Given node_id, returns list of tactics to try.
            prior_generator: Given node_id, returns prior probabilities.
        """
        iteration = 0
        while True:
            node_id = self.select_node()
            if node_id is None:
                logger.debug(f"Search exhausted: no more open nodes "
                             f"(expanded={self._stats.nodes_expanded})")
                break

            tactics = tactic_generator(node_id)
            priors = (prior_generator(node_id) if prior_generator
                      else [0.5] * len(tactics))

            results = self.try_batch(node_id, tactics, priors)

            if any(r.is_complete for r in results):
                logger.info(f"Search solved in {self._stats.nodes_expanded} "
                            f"expansions, depth={self._stats.solution_depth}")
                break

            iteration += 1
            if iteration % 50 == 0:
                logger.debug(
                    f"Search progress: {self._stats.nodes_expanded} expanded, "
                    f"{len(self._tree.open_leaves())} open, "
                    f"max_depth={self._stats.max_depth_reached}")

        self._stats.time_ms = (time.perf_counter() - self._start_time) * 1000
        return self._stats

    def stats(self) -> dict:
        s = self._stats
        return {
            "total_nodes": self._tree.size(),
            "nodes_expanded": s.nodes_expanded,
            "nodes_pruned": s.nodes_pruned,
            "l0_filtered": s.l0_filtered,
            "l1_filtered": s.l1_filtered,
            "max_depth": s.max_depth_reached,
            "open_leaves": len(self._tree.open_leaves()),
            "is_solved": s.is_solved,
            "solution_depth": s.solution_depth,
            "solution_path": s.solution_path,
            "time_ms": s.time_ms,
        }

    # ── Node Selection Strategies ──

    def _select_best_first(self) -> Optional[int]:
        """Select highest-scored node from frontier."""
        while self._frontier:
            neg_score, node_id = heapq.heappop(self._frontier)
            node = self._tree.get(NodeId(node_id))
            if node and node.status == NodeStatus.OPEN:
                if node.depth < self._config.max_depth:
                    return node_id
                self._stats.nodes_pruned += 1
        return None

    def _select_ucb(self) -> Optional[int]:
        """Select node using UCB1 formula (MCTS-style).

        UCB1(node) = exploitation + c * sqrt(ln(N_parent) / n_i) + prior_bonus
        Virtual loss is subtracted from exploitation to discourage re-selection
        of nodes currently being explored by other workers.

        where:
          exploitation = (success_count - virtual_loss) / visit_count
          N_parent = parent node's visit count
          n_i = this node's visit count
          c = exploration constant
        """
        leaves = self._tree.open_leaves()
        if not leaves:
            return None

        c = self._config.ucb_c

        best_id = None
        best_ucb = -float('inf')

        for nid in leaves:
            node = self._tree.get(nid)
            if not node or node.depth >= self._config.max_depth:
                continue

            n_i = max(1, node.visit_count)
            w_i = node.success_count

            # Subtract virtual loss from win count
            vl = self._virtual_losses.get(nid.id, 0)
            exploitation = (w_i - vl) / n_i

            # Get parent's visit count for the exploration term
            parent_visits = 1
            if node.parent is not None:
                parent_node = self._tree.get(node.parent)
                if parent_node:
                    parent_visits = max(1, parent_node.visit_count)

            exploration = c * math.sqrt(math.log(parent_visits + 1) / n_i)

            # Add prior bonus
            prior = self._priors.get(nid.id, 0.5)
            prior_bonus = self._config.prior_weight * prior / (1 + n_i)

            ucb = exploitation + exploration + prior_bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_id = nid.id

        return best_id

    def _select_bfs(self) -> Optional[int]:
        """BFS: select shallowest open leaf."""
        leaves = self._tree.open_leaves()
        if not leaves:
            return None
        # Sort by depth, take shallowest
        best = min(leaves,
                   key=lambda nid: self._tree.get(nid).depth
                   if self._tree.get(nid) else float('inf'))
        node = self._tree.get(best)
        if node and node.depth < self._config.max_depth:
            return best.id
        return None

    def _select_dfs(self) -> Optional[int]:
        """DFS: select deepest open leaf."""
        leaves = self._tree.open_leaves()
        if not leaves:
            return None
        best = max(leaves,
                   key=lambda nid: self._tree.get(nid).depth
                   if self._tree.get(nid) else 0)
        node = self._tree.get(best)
        if node and node.depth < self._config.max_depth:
            return best.id
        return None

    # ── Backpropagation ──

    def _backpropagate(self, node_id: int, success: bool):
        """Propagate result back up the tree, updating visit counts.

        Thread-safe: acquires self._lock to prevent concurrent update loss
        when multiple workers call _backpropagate simultaneously.

        Uses SearchTree.backpropagate() which returns a new immutable tree
        rather than mutating _nodes in place — preserving PMap semantics.
        """
        with self._lock:
            self._tree = self._tree.backpropagate(
                NodeId(node_id), success)

    # ── Scoring ──

    def _compute_score(self, node_id: int, result: TacticResult,
                       prior: float, depth: int) -> float:
        """Compute heuristic score for a node.

        Higher = more promising. All weights are configurable via SearchConfig.
        """
        cfg = self._config
        score = prior * cfg.prior_weight

        if result.goals_before > 0:
            reduction = (result.goals_before - result.goals_after) / result.goals_before
            score += reduction * cfg.goal_reduction_weight

        if result.goals_after == 0:
            score += cfg.completion_bonus

        score -= depth * cfg.depth_penalty

        if result.elapsed_us > 0:
            score += self._speed_bonus(result.elapsed_us)

        return score

    @staticmethod
    def _speed_bonus_fn(elapsed_us: float, threshold_us: float,
                        weight: float) -> float:
        """Log-based speed bonus that works across local (~1μs) and REPL (~50ms) scales.

        Returns weight * max(0, 1 - log2(elapsed/threshold + 1)).
        Examples (threshold=500000μs=500ms):
          1μs    → ~0.10  (fast local tactic)
          100μs  → ~0.10  (slow local tactic)
          50ms   → ~0.04  (fast REPL tactic)
          200ms  → ~0.02  (medium REPL tactic)
          500ms  → 0.00   (slow REPL tactic, at threshold)
        """
        if threshold_us <= 0 or elapsed_us <= 0:
            return 0.0
        ratio = elapsed_us / threshold_us
        bonus = max(0.0, 1.0 - math.log2(ratio + 1))
        return bonus * weight

    def _speed_bonus(self, elapsed_us: float) -> float:
        cfg = self._config
        return self._speed_bonus_fn(
            elapsed_us, cfg.speed_bonus_threshold_us, cfg.speed_bonus_weight)

    # ── Path extraction ──

    def _extract_path(self, node_id: int) -> list[str]:
        """Extract the tactic path from root to the given node."""
        path = []
        current = NodeId(node_id)
        while current is not None:
            node = self._tree.get(current)
            if node is None:
                break
            if node.tactic:
                path.append(node.tactic)
            current = node.parent
        path.reverse()
        return path
