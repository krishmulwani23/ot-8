"""
Register Allocation Formulated as an Optimization Problem (Track F):
Graph-Coloring Register Allocation solved via Integer Linear Programming (ILP)
and compared with Chaitin's Greedy heuristic.
"""

import time
import numpy as np
from typing import List, Dict, Set, Tuple, Any
from scipy.optimize import milp, LinearConstraint, Bounds
from track_f_compiler.ir import Instruction

class InterferenceGraph:
    """Computes variable live ranges and constructs interference graph."""
    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions
        self.variables: List[str] = []
        self.live_ranges: Dict[str, Tuple[int, int]] = {}
        self.interferences: Set[Tuple[str, str]] = set()
        self._analyze_liveness()

    def _analyze_liveness(self):
        # Determine first def and last use of each variable
        first_def = {}
        last_use = {}

        for idx, inst in enumerate(self.instructions):
            for d in inst.defs():
                if d not in first_def:
                    first_def[d] = idx
                last_use[d] = max(last_use.get(d, idx), idx)
            for u in inst.uses():
                if u not in first_def:
                    first_def[u] = 0
                last_use[u] = max(last_use.get(u, idx), idx)

        all_vars = sorted(list(set(first_def.keys()) | set(last_use.keys())))
        self.variables = all_vars

        for v in all_vars:
            self.live_ranges[v] = (first_def.get(v, 0), last_use.get(v, 0))

        # Two variables interfere if their live intervals overlap
        for i in range(len(all_vars)):
            for j in range(i + 1, len(all_vars)):
                v1, v2 = all_vars[i], all_vars[j]
                s1, e1 = self.live_ranges[v1]
                s2, e2 = self.live_ranges[v2]
                if not (e1 < s2 or e2 < s1):
                    self.interferences.add((v1, v2))


class ILPRegisterAllocator:
    """
    Formulates Register Allocation as an exact Integer Linear Program (ILP):
    Given K physical registers:
    Minimize: sum_v SpillWeight(v) * s_v
    Subject to:
      sum_k x_{v, k} + s_v = 1 (each var in a register or spilled to RAM)
      x_{u, k} + x_{v, k} <= 1 for all (u, v) in Interferences, for each register k
      x_{v, k}, s_v in {0, 1}
    """
    def __init__(self, k_registers: int = 3):
        self.K = k_registers

    def allocate(self, ig: InterferenceGraph) -> Dict[str, Any]:
        t0 = time.perf_counter()
        vars_list = ig.variables
        V = len(vars_list)
        K = self.K

        if V == 0:
            return {"allocation": {}, "spills": [], "runtime_ms": 0.0}

        var_to_idx = {v: i for i, v in enumerate(vars_list)}

        # Total binary variables:
        # x_{v, k}: V * K variables
        # s_v: V spill variables
        # Total: V * (K + 1)
        n_vars = V * (K + 1)

        def x_idx(v_i, k):
            return v_i * K + k

        def s_idx(v_i):
            return V * K + v_i

        # Objective: minimize spill cost
        c = np.zeros(n_vars, dtype=np.float64)
        for v_i in range(V):
            # Spill cost weighted by frequency of variable usage
            c[s_idx(v_i)] = 10.0 # memory spill penalty

        # Constraints
        # 1. Assignment: sum_k x_{v, k} + s_v = 1  (V constraints)
        # 2. Interference: x_{u, k} + x_{v, k} <= 1 (len(interferences) * K constraints)
        interferences = list(ig.interferences)
        n_interf = len(interferences)
        n_constraints = V + n_interf * K

        A = np.zeros((n_constraints, n_vars), dtype=np.float64)
        lhs = np.zeros(n_constraints, dtype=np.float64)
        rhs = np.zeros(n_constraints, dtype=np.float64)

        row = 0
        for v_i in range(V):
            for k in range(K):
                A[row, x_idx(v_i, k)] = 1.0
            A[row, s_idx(v_i)] = 1.0
            lhs[row] = 1.0
            rhs[row] = 1.0
            row += 1

        for u, v in interferences:
            u_i = var_to_idx[u]
            v_i = var_to_idx[v]
            for k in range(K):
                A[row, x_idx(u_i, k)] = 1.0
                A[row, x_idx(v_i, k)] = 1.0
                lhs[row] = -np.inf
                rhs[row] = 1.0
                row += 1

        integrality = np.ones(n_vars, dtype=int) # all variables are binary
        bounds = Bounds(lb=0.0, ub=1.0)
        constraints = LinearConstraint(A=A, lb=lhs, ub=rhs)

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        allocation = {}
        spills = []
        if res.success:
            sol = np.round(res.x).astype(int)
            for v_i, v_name in enumerate(vars_list):
                if sol[s_idx(v_i)] == 1:
                    spills.append(v_name)
                else:
                    for k in range(K):
                        if sol[x_idx(v_i, k)] == 1:
                            allocation[v_name] = f"R{k+1}"
                            break

        return {
            "solver": "ILP_Register_Allocation",
            "k_registers": K,
            "allocation": allocation,
            "spills": spills,
            "spill_count": len(spills),
            "status": res.status,
            "runtime_ms": elapsed_ms
        }
