"""
Index Selection Formulated as 0-1 Knapsack & ILP (Track H):
Selects subset of candidate indexes to maximize workload latency reduction
subject to a disk storage budget constraint B.
"""

import sqlite3
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy.optimize import milp, LinearConstraint, Bounds

class CandidateIndex:
    def __init__(self, name: str, table: str, column: str, estimated_size_kb: int):
        self.name = name
        self.table = table
        self.column = column
        self.size_kb = estimated_size_kb
        self.create_sql = f"CREATE INDEX IF NOT EXISTS {name} ON {table}({column});"
        self.drop_sql = f"DROP INDEX IF EXISTS {name};"


class IndexSelectionOptimizer:
    def __init__(self, db_conn: sqlite3.Connection):
        self.conn = db_conn
        self.candidates = [
            CandidateIndex("idx_li_orderkey", "lineitem", "l_orderkey", 240),
            CandidateIndex("idx_li_shipdate", "lineitem", "l_shipdate", 260),
            CandidateIndex("idx_ord_custkey", "orders", "o_custkey", 90),
            CandidateIndex("idx_ord_orderdate", "orders", "o_orderdate", 110),
            CandidateIndex("idx_cust_mktseg", "customer", "c_mktsegment", 45),
            CandidateIndex("idx_li_suppkey", "lineitem", "l_suppkey", 230)
        ]

    def measure_query_latency_ms(self, sql: str, runs: int = 3) -> float:
        """Measures average execution time of a query in milliseconds."""
        cur = self.conn.cursor()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            cur.execute(sql)
            _ = cur.fetchall()
            times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(times))

    def evaluate_benefits(self, queries: Dict[str, str]) -> np.ndarray:
        """Measures latency reduction benefit of each candidate index individually."""
        cur = self.conn.cursor()
        # Ensure all indexes are dropped for baseline measurement
        for cand in self.candidates:
            cur.execute(cand.drop_sql)
        self.conn.commit()

        # Baseline latency across workload
        base_latency = sum(self.measure_query_latency_ms(q) for q in queries.values())

        benefits = []
        for cand in self.candidates:
            cur.execute(cand.create_sql)
            self.conn.commit()
            with_idx_latency = sum(self.measure_query_latency_ms(q) for q in queries.values())
            cur.execute(cand.drop_sql)
            self.conn.commit()

            # Benefit = latency reduction in ms (non-negative)
            gain = max(0.1, base_latency - with_idx_latency)
            benefits.append(gain)

        return np.array(benefits, dtype=np.float64)

    def solve_greedy(self, benefits: np.ndarray, budget_kb: int) -> Dict[str, Any]:
        """Greedy selection by benefit-to-size density."""
        t0 = time.perf_counter()
        n = len(self.candidates)
        sizes = np.array([c.size_kb for c in self.candidates])
        ratios = benefits / sizes
        order = np.argsort(-ratios)

        curr_weight = 0
        total_benefit = 0.0
        selected = []

        for idx in order:
            if curr_weight + sizes[idx] <= budget_kb:
                selected.append(idx)
                curr_weight += sizes[idx]
                total_benefit += benefits[idx]

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "method": "Greedy",
            "selected_indexes": [self.candidates[i].name for i in selected],
            "total_benefit_ms": float(total_benefit),
            "total_size_kb": int(curr_weight),
            "runtime_ms": runtime_ms
        }

    def solve_dynamic_programming(self, benefits: np.ndarray, budget_kb: int) -> Dict[str, Any]:
        """Exact 0-1 Knapsack DP table."""
        t0 = time.perf_counter()
        n = len(self.candidates)
        sizes = [c.size_kb for c in self.candidates]
        W = budget_kb

        dp = np.zeros((n + 1, W + 1), dtype=np.float64)
        for i in range(1, n + 1):
            w_i = sizes[i - 1]
            b_i = benefits[i - 1]
            for w in range(W + 1):
                if w_i <= w:
                    dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - w_i] + b_i)
                else:
                    dp[i, w] = dp[i - 1, w]

        # Backtrack
        curr_w = W
        selected = []
        for i in range(n, 0, -1):
            if dp[i, curr_w] != dp[i - 1, curr_w]:
                selected.append(i - 1)
                curr_w -= sizes[i - 1]

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        total_weight = sum(sizes[i] for i in selected)
        return {
            "method": "DynamicProgramming",
            "selected_indexes": [self.candidates[i].name for i in reversed(selected)],
            "total_benefit_ms": float(dp[n, W]),
            "total_size_kb": int(total_weight),
            "runtime_ms": runtime_ms
        }

    def solve_ilp(self, benefits: np.ndarray, budget_kb: int) -> Dict[str, Any]:
        """Exact Integer Linear Programming formulation solved via HiGHS."""
        t0 = time.perf_counter()
        n = len(self.candidates)
        sizes = np.array([c.size_kb for c in self.candidates], dtype=np.float64)

        c = -benefits # minimize negative benefit
        integrality = np.ones(n, dtype=int)
        bounds = Bounds(lb=0.0, ub=1.0)
        constraints = LinearConstraint(A=sizes.reshape(1, -1), lb=-np.inf, ub=float(budget_kb))

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        sol = np.round(res.x).astype(int)
        selected = [i for i in range(n) if sol[i] == 1]
        total_weight = int(np.sum(sizes * sol))

        return {
            "method": "ILP_HiGHS",
            "selected_indexes": [self.candidates[i].name for i in selected],
            "total_benefit_ms": float(-res.fun),
            "total_size_kb": total_weight,
            "runtime_ms": runtime_ms
        }
