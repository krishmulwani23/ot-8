"""
Workflow Engine - Directed Acyclic Graph (DAG) Module (Track J).
Provides graph topological sort, dependency validation, cycle detection,
and critical path analysis for enterprise task orchestration.
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque

class DAGValidationError(Exception):
    """Raised when DAG violates acyclic or dependency invariants."""
    pass


class Node:
    """Represents an execution unit within the DAG."""
    def __init__(self, node_id: str, estimated_duration: float = 1.0, priority: int = 0):
        self.node_id = node_id
        self.estimated_duration = estimated_duration
        self.priority = priority
        self.dependencies: Set[str] = set()
        self.dependents: Set[str] = set()
        self.metadata: Dict[str, str] = {}

    def add_dependency(self, parent_id: str):
        self.dependencies.add(parent_id)

    def add_dependent(self, child_id: str):
        self.dependents.add(child_id)


class DAG:
    """Directed Acyclic Graph orchestrator for task workflows."""
    def __init__(self, dag_id: str):
        self.dag_id = dag_id
        self.nodes: Dict[str, Node] = {}
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)

    def add_node(self, node_id: str, estimated_duration: float = 1.0, priority: int = 0) -> Node:
        if node_id in self.nodes:
            return self.nodes[node_id]
        node = Node(node_id, estimated_duration, priority)
        self.nodes[node_id] = node
        return node

    def add_edge(self, parent_id: str, child_id: str):
        if parent_id not in self.nodes:
            self.add_node(parent_id)
        if child_id not in self.nodes:
            self.add_node(child_id)

        self.nodes[parent_id].add_dependent(child_id)
        self.nodes[child_id].add_dependency(parent_id)
        self._adjacency[parent_id].add(child_id)
        self._reverse_adjacency[child_id].add(parent_id)

    def detect_cycles(self) -> bool:
        """Kahn's algorithm for cycle detection."""
        in_degree = {nid: len(node.dependencies) for nid, node in self.nodes.items()}
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for child in self._adjacency[curr]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return visited_count != len(self.nodes)

    def topological_sort(self) -> List[str]:
        """Returns valid topological execution sequence."""
        if self.detect_cycles():
            raise DAGValidationError(f"DAG '{self.dag_id}' contains cycles!")

        in_degree = {nid: len(node.dependencies) for nid, node in self.nodes.items()}
        queue = deque(sorted([nid for nid, deg in in_degree.items() if deg == 0]))
        order = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for child in sorted(list(self._adjacency[curr])):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def compute_critical_path(self) -> Tuple[List[str], float]:
        """Calculates critical path and makespan using dynamic programming."""
        order = self.topological_sort()
        earliest_start: Dict[str, float] = {nid: 0.0 for nid in self.nodes}
        earliest_finish: Dict[str, float] = {}
        predecessor_on_cp: Dict[str, Optional[str]] = {nid: None for nid in self.nodes}

        for nid in order:
            node = self.nodes[nid]
            finish = earliest_start[nid] + node.estimated_duration
            earliest_finish[nid] = finish

            for child in self._adjacency[nid]:
                if finish > earliest_start[child]:
                    earliest_start[child] = finish
                    predecessor_on_cp[child] = nid

        if not earliest_finish:
            return [], 0.0

        end_node = max(earliest_finish.items(), key=lambda kv: kv[1])[0]
        makespan = earliest_finish[end_node]

        path = []
        curr = end_node
        while curr is not None:
            path.append(curr)
            curr = predecessor_on_cp[curr]
        path.reverse()
        return path, makespan
