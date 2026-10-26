"""
Discrete Optimization Solvers (Track B):
1. Greedy Heuristic
2. Dynamic Programming (Exact 0-1)
3. Branch-and-Bound (Tree Search with LP Relaxation Bound & Pruning)
4. Integer Linear Programming (ILP) via HiGHS Solver
"""

import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import milp, LinearConstraint, Bounds

class KnapsackInstance:
    """Represents a 0-1 Knapsack / Combinatorial Resource Allocation Problem."""
    def __init__(self, values: np.ndarray, weights: np.ndarray, capacity: float):
        self.values = np.asarray(values, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.capacity = float(capacity)
        self.n = len(values)
        assert len(weights) == self.n, "Values and weights must have equal length"


class GreedyKnapsackSolver:
    """
    Greedy heuristic: sorts items by efficiency ratio (value / weight).
    Takes items until capacity is exhausted.
    Complexity: O(n log n).
    """
    def solve(self, instance: KnapsackInstance) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ratios = instance.values / (instance.weights + 1e-12)
        sorted_indices = np.argsort(-ratios)

        current_weight = 0.0
        total_value = 0.0
        selection = np.zeros(instance.n, dtype=int)

        for idx in sorted_indices:
            w = instance.weights[idx]
            if current_weight + w <= instance.capacity:
                selection[idx] = 1
                current_weight += w
                total_value += instance.values[idx]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "solver": "Greedy",
            "total_value": float(total_value),
            "total_weight": float(current_weight),
            "selection": selection.tolist(),
            "runtime_ms": elapsed_ms,
            "is_exact": False
        }


class DynamicProgrammingKnapsackSolver:
    """
    Exact Dynamic Programming table solver for integer weights.
    Table: DP[i, w] = max value using subset of first i items with weight limit w.
    Complexity: O(n * W) pseudo-polynomial.
    """
    def solve(self, instance: KnapsackInstance) -> Dict[str, Any]:
        t0 = time.perf_counter()
        W = int(np.floor(instance.capacity))
        int_weights = np.round(instance.weights).astype(int)
        values = instance.values
        n = instance.n

        # DP table: (n + 1) x (W + 1)
        dp = np.zeros((n + 1, W + 1), dtype=np.float64)

        for i in range(1, n + 1):
            w_i = int_weights[i - 1]
            v_i = values[i - 1]
            for w in range(W + 1):
                if w_i <= w:
                    dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - w_i] + v_i)
                else:
                    dp[i, w] = dp[i - 1, w]

        # Backtrack optimal selection
        selection = np.zeros(n, dtype=int)
        curr_w = W
        for i in range(n, 0, -1):
            if dp[i, curr_w] != dp[i - 1, curr_w]:
                selection[i - 1] = 1
                curr_w -= int_weights[i - 1]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_weight = float(np.sum(instance.weights * selection))
        return {
            "solver": "DynamicProgramming",
            "total_value": float(dp[n, W]),
            "total_weight": total_weight,
            "selection": selection.tolist(),
            "runtime_ms": elapsed_ms,
            "is_exact": True
        }


class BranchAndBoundKnapsackSolver:
    """
    Exact Branch-and-Bound tree search.
    Computes upper bound via Fractional Knapsack relaxation at each node.
    Prunes subtrees where upper bound <= best value found so far.
    """
    def solve(self, instance: KnapsackInstance) -> Dict[str, Any]:
        t0 = time.perf_counter()
        n = instance.n
        # Sort items by value density (v_i / w_i) descending
        ratios = instance.values / (instance.weights + 1e-12)
        order = np.argsort(-ratios)
        v_sorted = instance.values[order]
        w_sorted = instance.weights[order]
        cap = instance.capacity

        def fractional_bound(level: int, current_w: float, current_v: float) -> float:
            """Compute linear relaxation upper bound."""
            if current_w > cap:
                return 0.0
            bound = current_v
            rem_cap = cap - current_w
            for i in range(level, n):
                if w_sorted[i] <= rem_cap:
                    rem_cap -= w_sorted[i]
                    bound += v_sorted[i]
                else:
                    bound += v_sorted[i] * (rem_cap / w_sorted[i])
                    break
            return bound

        best_value = 0.0
        best_selection_sorted = np.zeros(n, dtype=int)
        nodes_explored = 0

        # Stack contains (level, current_weight, current_value, selection_prefix)
        # Using DFS with best-first expansion priority
        stack = [(0, 0.0, 0.0, [])]

        while stack:
            level, curr_w, curr_v, prefix = stack.pop()
            nodes_explored += 1

            if curr_w <= cap and curr_v > best_value:
                best_value = curr_v
                # Pad prefix with zeros for unassigned items
                best_selection_sorted = np.array(prefix + [0] * (n - len(prefix)), dtype=int)

            if level == n or curr_w > cap:
                continue

            # Check if this node can be pruned
            ub = fractional_bound(level, curr_w, curr_v)
            if ub <= best_value:
                continue # PRUNE branch!

            # Branch 1: Exclude item level (0)
            stack.append((level + 1, curr_w, curr_v, prefix + [0]))

            # Branch 2: Include item level (1) if within capacity
            if curr_w + w_sorted[level] <= cap:
                stack.append((level + 1, curr_w + w_sorted[level], curr_v + v_sorted[level], prefix + [1]))

        # Map sorted selection back to original item indices
        final_selection = np.zeros(n, dtype=int)
        for i in range(n):
            if best_selection_sorted[i] == 1:
                final_selection[order[i]] = 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "solver": "BranchAndBound",
            "total_value": float(best_value),
            "total_weight": float(np.sum(instance.weights * final_selection)),
            "selection": final_selection.tolist(),
            "nodes_explored": nodes_explored,
            "runtime_ms": elapsed_ms,
            "is_exact": True
        }


class ILPKnapsackSolver:
    """
    Exact Integer Linear Programming (ILP) formulation:
    Maximize: c^T x
    Subject to: A x <= b
    x_i in {0, 1}
    Solved using HiGHS Branch-and-Cut engine.
    """
    def solve(self, instance: KnapsackInstance) -> Dict[str, Any]:
        t0 = time.perf_counter()
        n = instance.n
        # scipy.optimize.milp minimizes c^T x, so c = -values
        c = -instance.values
        integrality = np.ones(n, dtype=int) # all variables are binary integers
        bounds = Bounds(lb=0.0, ub=1.0)
        constraints = LinearConstraint(A=instance.weights.reshape(1, -1), lb=-np.inf, ub=instance.capacity)

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        selection = np.round(res.x).astype(int)
        return {
            "solver": "ILP_HiGHS",
            "total_value": float(-res.fun),
            "total_weight": float(np.sum(instance.weights * selection)),
            "selection": selection.tolist(),
            "status": res.status,
            "success": res.success,
            "runtime_ms": elapsed_ms,
            "is_exact": True
        }
