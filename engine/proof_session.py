"""engine/proof_session.py — 证明会话管理器

管理证明过程中 env_id 构成的状态树, 支持:
  1. 快照/回退: 在任意 env_id 处 fork, 尝试不同 tactic
  2. 跨会话复用: 预热环境 (import + 常用引理) 的 env_id 持久化
  3. 增量验证: 只重验从分叉点开始变更的步骤

Lean4 REPL 的 env_id 语义:
  - 每条命令执行后返回一个新的 env_id
  - 新命令可以引用任意旧 env_id, 实现零成本 fork
  - env_id 是不可变快照, 引用旧 env_id 不影响已有状态

本模块在此基础上构建状态树:
  env_0 (import Mathlib)
    ├── env_1 (theorem header)
    │   ├── env_2 (tactic: intro n)
    │   │   ├── env_3 (tactic: simp)     ← 失败, 回退到 env_2
    │   │   └── env_4 (tactic: omega)    ← 成功, 继续
    │   │       └── env_5 (tactic: rfl)  ← 证明完成
    │   └── env_6 (tactic: cases n)      ← 从 env_1 fork 的替代路径
    └── env_7 (不同 theorem)             ← 复用 import 环境

Usage::

    async with ProofSessionManager(pool) as mgr:
        # 开始新证明
        session = await mgr.begin_proof("theorem t : 1+1=2 := by")

        # 尝试 tactic
        result = await session.try_step("norm_num")
        if result.success:
            print(f"Proof complete at env_id={result.env_id}")

        # 回退并尝试不同路径
        await session.rewind(steps=1)
        result = await session.try_step("simp")
"""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from engine.lean_pool import TacticFeedback

logger = logging.getLogger(__name__)


@dataclass
class EnvNode:
    """env_id 状态树中的节点"""
    env_id: int
    parent_env_id: int = -1
    tactic: str = ""                   # 产生此节点的 tactic
    goals: list[str] = field(default_factory=list)
    is_proof_complete: bool = False
    children: list[int] = field(default_factory=list)  # 子节点的 env_id
    created_at: float = field(default_factory=time.time)
    depth: int = 0                     # 从 root 到此节点的步数

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class ProofSessionState:
    """单个证明的完整状态"""
    theorem: str
    root_env_id: int                   # import 之后的基础环境
    theorem_env_id: int                # theorem header 之后的环境
    current_env_id: int                # 当前探索位置
    nodes: dict[int, EnvNode] = field(default_factory=dict)
    tactic_history: list[str] = field(default_factory=list)
    best_depth: int = 0                # 达到过的最大深度
    solved: bool = False
    proof_path: list[int] = field(default_factory=list)  # 成功路径的 env_id 序列


class ProofSession:
    """单个证明的交互式会话

    封装在一个定理上的状态树操作。
    所有 try_step / rewind / fork 操作通过 AsyncLeanPool 执行。
    """

    def __init__(self, state: ProofSessionState, pool: 'AsyncLeanPool'):
        self._state = state
        self._pool = pool

    @property
    def current_env_id(self) -> int:
        return self._state.current_env_id

    @property
    def current_depth(self) -> int:
        node = self._state.nodes.get(self._state.current_env_id)
        return node.depth if node else 0

    @property
    def current_goals(self) -> list[str]:
        node = self._state.nodes.get(self._state.current_env_id)
        return node.goals if node else []

    @property
    def is_solved(self) -> bool:
        return self._state.solved

    @property
    def tactic_history(self) -> list[str]:
        return list(self._state.tactic_history)

    async def try_step(self, tactic: str) -> TacticFeedback:
        """在当前 env_id 上执行一条 tactic, 前进一步

        成功: current_env_id 更新为新 env_id, 树中添加子节点
        失败: current_env_id 不变, 返回错误信息
        """
        result = await self._pool.try_tactic(
            self._state.current_env_id, tactic)

        if result.success:
            new_env = result.new_env_id
            parent = self._state.current_env_id
            parent_node = self._state.nodes.get(parent)
            new_depth = (parent_node.depth + 1) if parent_node else 1

            # 添加新节点
            node = EnvNode(
                env_id=new_env,
                parent_env_id=parent,
                tactic=tactic,
                goals=result.remaining_goals,
                is_proof_complete=result.is_proof_complete,
                depth=new_depth)
            self._state.nodes[new_env] = node

            # 更新父节点的 children
            if parent_node:
                parent_node.children.append(new_env)

            # 更新当前位置
            self._state.current_env_id = new_env
            self._state.tactic_history.append(tactic)
            self._state.best_depth = max(
                self._state.best_depth, new_depth)

            if result.is_proof_complete:
                self._state.solved = True
                self._state.proof_path = self._trace_path(new_env)
                logger.info(
                    f"Proof complete! Path: {len(self._state.proof_path)} steps")

        return result

    async def try_steps(self, tactics: list[str]) -> list[TacticFeedback]:
        """顺序执行多条 tactic, 在第一条失败时停止"""
        results = []
        for tactic in tactics:
            result = await self.try_step(tactic)
            results.append(result)
            if not result.success:
                break
        return results

    async def try_alternatives(self, tactics: list[str]) -> list[TacticFeedback]:
        """在当前 env_id 上并行尝试多条 tactic (不改变 current)

        每条 tactic 都从当前 env_id fork, 互不影响。
        用于 MCTS 式的宽度搜索。
        """
        env_id = self._state.current_env_id
        results = await self._pool.try_tactics_parallel(env_id, tactics)

        # 将成功的添加为子节点 (但不改变 current)
        parent_node = self._state.nodes.get(env_id)
        for result in results:
            if result.success:
                new_depth = (parent_node.depth + 1) if parent_node else 1
                node = EnvNode(
                    env_id=result.new_env_id,
                    parent_env_id=env_id,
                    tactic=result.tactic,
                    goals=result.remaining_goals,
                    is_proof_complete=result.is_proof_complete,
                    depth=new_depth)
                self._state.nodes[result.new_env_id] = node
                if parent_node:
                    parent_node.children.append(result.new_env_id)

        return results

    def rewind(self, steps: int = 1) -> int:
        """回退 N 步, 返回回退后的 env_id

        不删除节点 — 回退只是移动 current 指针,
        旧节点仍然可用 (Lean4 env_id 是不可变快照)。

        回退后从实际路径重建 tactic_history, 保证一致性。
        """
        current = self._state.current_env_id
        for _ in range(steps):
            node = self._state.nodes.get(current)
            if not node or node.parent_env_id < 0:
                break
            current = node.parent_env_id

        self._state.current_env_id = current

        # 从实际路径重建 tactic_history (与 goto 保持一致)
        path = self._trace_path(current)
        self._state.tactic_history = [
            self._state.nodes[eid].tactic
            for eid in path
            if self._state.nodes[eid].tactic]

        return current

    def goto(self, env_id: int) -> bool:
        """跳转到任意已知 env_id

        用于探索树中的不同分支。
        """
        if env_id in self._state.nodes:
            self._state.current_env_id = env_id
            # 重建 tactic_history
            self._state.tactic_history = [
                self._state.nodes[eid].tactic
                for eid in self._trace_path(env_id)
                if self._state.nodes[eid].tactic]
            return True
        return False

    def get_successful_branches(self) -> list[list[str]]:
        """获取所有到达过的最深路径 (用于分析)"""
        leaves = [n for n in self._state.nodes.values() if n.is_leaf]
        leaves.sort(key=lambda n: -n.depth)
        paths = []
        for leaf in leaves[:5]:
            path_ids = self._trace_path(leaf.env_id)
            tactics = [self._state.nodes[eid].tactic
                       for eid in path_ids
                       if self._state.nodes[eid].tactic]
            paths.append(tactics)
        return paths

    def get_proof_script(self) -> str:
        """获取成功证明的 tactic 脚本"""
        if not self._state.solved or not self._state.proof_path:
            return ""
        tactics = [self._state.nodes[eid].tactic
                   for eid in self._state.proof_path
                   if self._state.nodes[eid].tactic]
        return ":= by\n  " + "\n  ".join(tactics)

    def tree_stats(self) -> dict:
        """状态树统计"""
        nodes = self._state.nodes
        return {
            "total_nodes": len(nodes),
            "max_depth": self._state.best_depth,
            "current_depth": self.current_depth,
            "solved": self._state.solved,
            "leaf_count": sum(1 for n in nodes.values() if n.is_leaf),
            "branch_factor": (
                len(nodes) / max(1, self._state.best_depth)),
        }

    def _trace_path(self, env_id: int) -> list[int]:
        """从 env_id 回溯到 root, 返回路径 (root → env_id)"""
        path = []
        visited = set()
        current = env_id
        while current >= 0 and current in self._state.nodes:
            if current in visited:
                break  # 防止自引用循环
            visited.add(current)
            path.append(current)
            current = self._state.nodes[current].parent_env_id
        path.reverse()
        return path


class ProofSessionManager:
    """证明会话管理器

    管理多个证明的状态树, 支持:
      - 跨证明的环境复用 (共享 import 预热环境)
      - 会话快照/恢复
      - 并行证明调度
    """

    def __init__(self, pool: 'AsyncLeanPool', store=None):
        self._pool = pool
        self._sessions: dict[str, ProofSession] = {}
        self._context_ids: dict[str, int] = {}  # session_id → store context_id
        self._store = store  # Optional[ProofContextStore]
        self._base_env_id: int = -1   # import 后的基础环境

    async def begin_proof(self, theorem: str,
                          session_id: str = "") -> ProofSession:
        """开始一个新的证明会话

        1. 复用 import 预热的基础 env_id
        2. 发送 theorem header 获取 theorem_env_id
        3. 返回 ProofSession
        """
        if not session_id:
            session_id = f"proof_{len(self._sessions)}_{time.time():.0f}"

        # 复用基础环境
        if self._base_env_id < 0:
            # 使用 pool 中任意会话的 base_env_id
            for s in self._pool._sessions:
                if s.is_alive and s.base_env_id >= 0:
                    self._base_env_id = s.base_env_id
                    break

        base = max(0, self._base_env_id)

        # 发送 theorem header
        result = await self._pool.try_tactic(base, theorem)
        theorem_env_id = result.new_env_id if result.success else base

        root_node = EnvNode(
            env_id=base, tactic="", goals=[], depth=0)

        nodes = {base: root_node}

        # 当 theorem_env_id == base (fallback 模式), 不创建自引用节点
        if theorem_env_id != base:
            theorem_node = EnvNode(
                env_id=theorem_env_id,
                parent_env_id=base,
                tactic=theorem,
                goals=result.remaining_goals if result.success else [],
                depth=1)
            nodes[theorem_env_id] = theorem_node

        state = ProofSessionState(
            theorem=theorem,
            root_env_id=base,
            theorem_env_id=theorem_env_id,
            current_env_id=theorem_env_id,
            nodes=nodes,
            tactic_history=[])

        session = ProofSession(state, self._pool)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ProofSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def remove_session(self, session_id: str):
        self._sessions.pop(session_id, None)
        self._context_ids.pop(session_id, None)

    async def save_session(self, session_id: str) -> Optional[int]:
        """Persist a proof session to the store. Returns context_id."""
        if not self._store:
            return None
        session = self._sessions.get(session_id)
        if not session:
            return None
        ctx_id = self._context_ids.get(session_id)
        ctx_id = await self._store.save(session._state, context_id=ctx_id)
        self._context_ids[session_id] = ctx_id
        return ctx_id

    async def load_session(self, context_id: int,
                           session_id: str = "") -> Optional[ProofSession]:
        """Resume a proof session from the store.

        Note: env_ids from the stored state are stale (they belonged to a
        previous REPL process). The loaded session preserves the proof tree
        structure and tactic history for analysis, but live REPL interaction
        requires replaying tactics from the root.
        """
        if not self._store:
            return None
        state = await self._store.load(context_id)
        if not state:
            return None
        if not session_id:
            session_id = f"resumed_{context_id}_{time.time():.0f}"
        session = ProofSession(state, self._pool)
        self._sessions[session_id] = session
        self._context_ids[session_id] = context_id
        return session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self._sessions.clear()
