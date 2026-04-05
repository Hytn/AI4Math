"""Persistent search tree for proof exploration."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pyrsistent import pmap, pvector

class NodeStatus(Enum):
    OPEN = "open"; SOLVED = "solved"; FAILED = "failed"; PRUNED = "pruned"

@dataclass(frozen=True)
class NodeId:
    id: int
    def __hash__(self): return hash(self.id)

@dataclass
class SearchNode:
    id: NodeId; state: object; parent: Optional[NodeId] = None
    tactic: Optional[str] = None; children: tuple = ()
    status: NodeStatus = NodeStatus.OPEN
    visit_count: int = 0; success_count: int = 0; depth: int = 0

class SearchTree:
    def __init__(self, initial_state):
        root_id = NodeId(0)
        root = SearchNode(root_id, initial_state)
        self._nodes = pmap({root_id: root})
        self._root = root_id; self._next = 1

    def expand(self, parent_id, tactic, new_state):
        child_id = NodeId(self._next)
        parent = self._nodes.get(parent_id)
        depth = parent.depth + 1 if parent else 0
        child = SearchNode(child_id, new_state, parent_id, tactic, depth=depth)
        new_nodes = self._nodes.set(child_id, child)
        if parent:
            updated = SearchNode(parent.id, parent.state, parent.parent, parent.tactic,
                                parent.children + (child_id,), parent.status,
                                parent.visit_count, parent.success_count, parent.depth)
            new_nodes = new_nodes.set(parent_id, updated)
        tree = SearchTree.__new__(SearchTree)
        tree._nodes = new_nodes; tree._root = self._root; tree._next = self._next + 1
        return tree, child_id

    def update_node(self, node_id: NodeId, *,
                    visit_count: int = None,
                    success_count: int = None,
                    status: NodeStatus = None) -> 'SearchTree':
        """Return a new SearchTree with the specified node fields updated.

        This is the proper way to modify node state — returns a new tree
        instance rather than mutating _nodes in place, preserving the
        immutable data structure semantics of PMap.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return self
        updated = SearchNode(
            node.id, node.state, node.parent, node.tactic,
            node.children,
            status if status is not None else node.status,
            visit_count if visit_count is not None else node.visit_count,
            success_count if success_count is not None else node.success_count,
            node.depth,
        )
        tree = SearchTree.__new__(SearchTree)
        tree._nodes = self._nodes.set(node_id, updated)
        tree._root = self._root
        tree._next = self._next
        return tree

    def backpropagate(self, node_id: NodeId, success: bool) -> 'SearchTree':
        """Propagate result up the tree, returning a new tree with updated counts.

        Traverses from node_id to root, incrementing visit_count on each node
        and success_count if success=True. Returns a new immutable tree.
        """
        nodes = self._nodes
        current_id = node_id
        while current_id is not None:
            node = nodes.get(current_id)
            if node is None:
                break
            updated = SearchNode(
                node.id, node.state, node.parent, node.tactic,
                node.children,
                NodeStatus.SOLVED if success else node.status,
                node.visit_count + 1,
                node.success_count + (1 if success else 0),
                node.depth,
            )
            nodes = nodes.set(current_id, updated)
            current_id = node.parent

        tree = SearchTree.__new__(SearchTree)
        tree._nodes = nodes
        tree._root = self._root
        tree._next = self._next
        return tree

    def get(self, nid): return self._nodes.get(nid)
    def root(self): return self._nodes[self._root]
    def size(self): return len(self._nodes)
    def open_leaves(self):
        return [n.id for n in self._nodes.values()
                if n.status == NodeStatus.OPEN and not n.children]
