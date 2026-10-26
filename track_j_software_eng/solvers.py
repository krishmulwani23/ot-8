"""
Test Suite Minimization & Prioritization Solvers (Track J):
Formulates test selection as an Integer Linear Programming (ILP) problem under time budget
and compares with Greedy heuristic and Naive sequential execution.
"""

import time
import numpy as np
from typing import List, Dict, Set, Any
from scipy.optimize import milp, LinearConstraint, Bounds
from track_j_software_eng.workflow_engine.test_suite import TestCaseMetadata

class TestSuiteOptimizer:
    def __init__(self, test_cases: List[TestCaseMetadata]):
        self.tests = test_cases
        self.n_tests = len(test_cases)
        # Collect all unique code blocks/branches
        all_blocks = set()
        for t in test_cases:
            all_blocks.update(t.covered_blocks)
        self.blocks = sorted(list(all_blocks))
        self.n_blocks = len(self.blocks)
        self.block_to_idx = {b: i for i, b in enumerate(self.blocks)}

    def naive_baseline(self) -> Dict[str, Any]:
        """Runs the entire test suite unconditionally."""
        t0 = time.perf_counter()
        total_time_ms = sum(t.duration_ms for t in self.tests)
        covered_blocks = set()
        total_fault = 0.0

        for t in self.tests:
            covered_blocks.update(t.covered_blocks)
            total_fault += t.fault_score

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "solver": "Naive_Full_Suite_Baseline",
            "selected_tests": [t.test_name for t in self.tests],
            "selected_count": self.n_tests,
            "total_execution_time_ms": total_time_ms,
            "covered_blocks_count": len(covered_blocks),
            "coverage_pct": 100.0,
            "fault_score": total_fault,
            "solve_time_ms": elapsed_ms
        }

    def solve_greedy(self, time_budget_ms: float) -> Dict[str, Any]:
        """Greedy heuristic: picks test with highest marginal (new coverage) per millisecond."""
        t0 = time.perf_counter()
        covered: Set[int] = set()
        selected: List[int] = []
        rem_budget = time_budget_ms
        unselected = set(range(self.n_tests))

        while unselected and rem_budget > 0:
            best_idx = -1
            best_ratio = -1.0

            for i in unselected:
                t = self.tests[i]
                if t.duration_ms <= rem_budget:
                    new_cov = len(t.covered_blocks - covered)
                    # Ratio: new coverage + fault weight per ms
                    ratio = (new_cov * 10.0 + t.fault_score) / max(0.1, t.duration_ms)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            covered.update(self.tests[best_idx].covered_blocks)
            rem_budget -= self.tests[best_idx].duration_ms
            unselected.remove(best_idx)

        total_exec_time = sum(self.tests[i].duration_ms for i in selected)
        total_fault = sum(self.tests[i].fault_score for i in selected)
        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "solver": "Greedy_Heuristic",
            "time_budget_ms": time_budget_ms,
            "selected_tests": [self.tests[i].test_name for i in selected],
            "selected_count": len(selected),
            "total_execution_time_ms": total_exec_time,
            "covered_blocks_count": len(covered),
            "coverage_pct": (len(covered) / self.n_blocks) * 100.0,
            "fault_score": total_fault,
            "solve_time_ms": solve_time_ms
        }

    def solve_ilp(self, time_budget_ms: float) -> Dict[str, Any]:
        """
        Exact Integer Linear Programming formulation:
        Variables:
          x_i in {0, 1} for i in [0, N-1]  (test selection)
          y_k in {0, 1} for k in [0, K-1]  (block coverage indicator)
        Maximize: sum_k (10 * y_k) + sum_i (fault_i * x_i)
        Subject to:
          sum_i (duration_i * x_i) <= time_budget_ms
          y_k - sum_{i: k in Cov(i)} x_i <= 0 for each block k
        """
        t0 = time.perf_counter()
        N = self.n_tests
        K = self.n_blocks
        n_vars = N + K

        # Objective vector c: scipy minimizes c^T z, so negate weights
        c = np.zeros(n_vars, dtype=np.float64)
        for i in range(N):
            c[i] = -self.tests[i].fault_score
        for k in range(K):
            c[N + k] = -10.0 # heavy weight on preserving code coverage

        # Constraints:
        # 1. Time budget: sum_i duration_i * x_i <= time_budget_ms  (1 constraint)
        # 2. Coverage linking: y_k - sum_{i: k in Cov(i)} x_i <= 0  (K constraints)
        n_constraints = 1 + K
        A = np.zeros((n_constraints, n_vars), dtype=np.float64)
        lhs = np.full(n_constraints, -np.inf, dtype=np.float64)
        rhs = np.zeros(n_constraints, dtype=np.float64)

        # 1. Budget constraint
        for i in range(N):
            A[0, i] = self.tests[i].duration_ms
        rhs[0] = time_budget_ms

        # 2. Coverage linking constraints
        for k_idx, block_id in enumerate(self.blocks):
            row = 1 + k_idx
            A[row, N + k_idx] = 1.0 # +y_k
            for i in range(N):
                if block_id in self.tests[i].covered_blocks:
                    A[row, i] = -1.0 # -x_i
            rhs[row] = 0.0

        integrality = np.ones(n_vars, dtype=int)
        bounds = Bounds(lb=0.0, ub=1.0)
        constraints = LinearConstraint(A=A, lb=lhs, ub=rhs)

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        solve_time_ms = (time.perf_counter() - t0) * 1000.0

        if not res.success:
            raise RuntimeError("ILP solver failed to find solution")

        sol = np.round(res.x).astype(int)
        selected_indices = [i for i in range(N) if sol[i] == 1]
        covered_indices = [k for k in range(K) if sol[N + k] == 1]

        total_exec_time = sum(self.tests[i].duration_ms for i in selected_indices)
        total_fault = sum(self.tests[i].fault_score for i in selected_indices)

        return {
            "solver": "ILP_HiGHS_Exact",
            "time_budget_ms": time_budget_ms,
            "selected_tests": [self.tests[i].test_name for i in selected_indices],
            "selected_count": len(selected_indices),
            "total_execution_time_ms": total_exec_time,
            "covered_blocks_count": len(covered_indices),
            "coverage_pct": (len(covered_indices) / self.n_blocks) * 100.0,
            "fault_score": total_fault,
            "solve_time_ms": solve_time_ms
        }
