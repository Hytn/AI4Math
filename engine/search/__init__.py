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
from dataclasses import dataclass, field
from typing import Optional, Callable
from engine.core import Expr, Name, MetaId
from engine.state import ProofState, SearchTree, NodeId, NodeStatus, GoalView
from engine.tactic import execute_tactic, TacticResult


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
                 config: SearchConfig = None):
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

    # ── Public API ──

    def goal_view(self, node_id: int) -> list[GoalView]:
        node = self._tree.get(NodeId(node_id))
        if not node:
            return []
        return [GoalView.from_goal(g, node.state) for g in node.state.goals()]

    def try_tactic(self, node_id: int, tactic_str: str,
                   prior: float = 0.5) -> ExpansionResult:
        """Try a single tactic at a node."""
        t0 = time.perf_counter_ns()
        node = self._tree.get(NodeId(node_id))
        if not node:
            return ExpansionResult(node_id, tactic_str, False,
                                  error={"kind": "not_found"}, elapsed_us=0)

        result = execute_tactic(node.state, tactic_str)
        elapsed = int((time.perf_counter_ns() - t0) / 1000)
        self._stats.nodes_expanded += 1

        if result.success:
            complete = result.state.is_complete()
            goals = [GoalView.from_goal(g, result.state)
                     for g in result.state.goals()]
            self._tree, child_id = self._tree.expand(
                NodeId(node_id), tactic_str, result.state)

            child_int = child_id.id
            self._stats.total_nodes = self._tree.size()
            depth = (self._tree.get(child_id).depth
                     if self._tree.get(child_id) else 0)
            self._stats.max_depth_reached = max(
                self._stats.max_depth_reached, depth)

            # Score the new node and add to frontier
            score = self._compute_score(child_int, result, prior, depth)
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
        while True:
            node_id = self.select_node()
            if node_id is None:
                break

            tactics = tactic_generator(node_id)
            priors = (prior_generator(node_id) if prior_generator
                      else [0.5] * len(tactics))

            results = self.try_batch(node_id, tactics, priors)

            if any(r.is_complete for r in results):
                break

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

        Correctly creates updated nodes via the persistent data structure
        instead of mutating the tree's internal PMap in place.
        """
        current_id = NodeId(node_id)
        updated_nodes = self._tree._nodes
        while current_id is not None:
            node = updated_nodes.get(current_id)
            if node is None:
                break

            # Create updated node with new visit/success counts
            new_visits = node.visit_count + 1
            new_success = node.success_count + (1 if success else 0)
            new_status = NodeStatus.SOLVED if success else node.status

            from engine.state.search_tree import SearchNode
            updated = SearchNode(
                node.id, node.state, node.parent, node.tactic,
                node.children, new_status,
                new_visits, new_success, node.depth)
            updated_nodes = updated_nodes.set(current_id, updated)

            current_id = node.parent

        # Replace the tree's nodes map atomically
        self._tree._nodes = updated_nodes

    # ── Scoring ──

    def _compute_score(self, node_id: int, result: TacticResult,
                       prior: float, depth: int) -> float:
        """Compute heuristic score for a node.

        Higher = more promising. Combines:
          - Prior probability from LLM (if available)
          - Goal reduction (fewer remaining goals = better)
          - Depth penalty (prefer shorter proofs)
          - Tactic speed (faster execution = simpler = better)
        """
        score = 0.0

        # Prior from LLM
        score += prior * self._config.prior_weight

        # Goal reduction bonus
        if result.goals_before > 0:
            reduction = (result.goals_before - result.goals_after) / result.goals_before
            score += reduction * 2.0

        # Completion bonus
        if result.goals_after == 0:
            score += 10.0

        # Depth penalty
        score -= depth * 0.05

        # Speed bonus (faster tactics are usually more likely correct)
        if result.elapsed_us > 0:
            speed_bonus = min(1.0, 100.0 / result.elapsed_us)
            score += speed_bonus * 0.1

        return score

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
