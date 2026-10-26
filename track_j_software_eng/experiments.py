"""
Experiments for Track J: Software-Engineering Optimization.
Benchmarks:
1. cProfile Profiling of test suite on >1000 LOC workflow engine codebase
2. Naive Full Suite vs Greedy Selection vs Exact ILP under CI time budgets
3. Trade-off analysis: CI Execution Speedup vs Code Coverage Preservation
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track_j_software_eng.workflow_engine.test_suite import get_all_test_cases
from track_j_software_eng.solvers import TestSuiteOptimizer
from track_j_software_eng.profiler import CodebaseProfiler


def run_swe_optimization_experiments():
    print("[Track J] Profiling Realistic Codebase (>1000 LOC Workflow Engine)...")
    prof_data = CodebaseProfiler.profile_full_suite()
    print(f"  Executed all {prof_data['total_tests_run']} tests under cProfile.")

    test_cases = get_all_test_cases()
    optimizer = TestSuiteOptimizer(test_cases)

    # 1. Naive Baseline (Full execution)
    res_naive = optimizer.naive_baseline()
    print(f"  Naive Baseline: {res_naive['selected_count']} tests, {res_naive['total_execution_time_ms']:.1f} ms, 100% coverage")

    # 2. Optimization under varying CI Time Deadlines
    budgets = [100.0, 150.0, 200.0, 250.0, 300.0]
    greedy_results = []
    ilp_results = []

    print("  Solving Test Selection across CI Time Budgets:")
    for b in budgets:
        g_sol = optimizer.solve_greedy(b)
        ilp_sol = optimizer.solve_ilp(b)

        greedy_results.append(g_sol)
        ilp_results.append(ilp_sol)

        print(f"    Budget {b:.0f} ms -> ILP Coverage: {ilp_sol['coverage_pct']:.1f}% ({ilp_sol['total_execution_time_ms']:.1f}ms) vs Greedy: {g_sol['coverage_pct']:.1f}% ({g_sol['total_execution_time_ms']:.1f}ms)")

    # Target deployment budget: 180 ms (Target: > 90% coverage with 2.5x speedup)
    target_budget = 180.0
    target_ilp = optimizer.solve_ilp(target_budget)
    target_speedup = res_naive["total_execution_time_ms"] / target_ilp["total_execution_time_ms"]

    print(f"  Target CI Pipeline Optimization (Budget: {target_budget} ms):")
    print(f"    Selected Tests:       {target_ilp['selected_count']} / {res_naive['selected_count']}")
    print(f"    CI Execution Latency: {res_naive['total_execution_time_ms']:.1f} ms -> {target_ilp['total_execution_time_ms']:.1f} ms")
    print(f"    Measured Speedup:     {target_speedup:.2f}x faster CI feedback!")
    print(f"    Coverage Retained:    {target_ilp['coverage_pct']:.1f}% of full suite coverage")

    # Plot Coverage vs Time Budget
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets, [s["coverage_pct"] for s in ilp_results], marker="o", label="Exact ILP (HiGHS)", color="#2ecc71", linewidth=2.2)
    ax.plot(budgets, [s["coverage_pct"] for s in greedy_results], marker="s", label="Greedy Heuristic", color="#e67e22", linewidth=2.0, linestyle="--")
    ax.axhline(100.0, color="gray", linestyle=":", label="100% Full Suite Coverage")

    ax.set_title("CI Test Suite Minimization: Code Coverage vs Execution Budget", fontsize=12, fontweight="bold")
    ax.set_xlabel("CI Time Budget Deadline (ms)", fontsize=10)
    ax.set_ylabel("Code Block / Branch Coverage (%)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_j_test_optimization.png", dpi=300)
    plt.close()
    print("  Saved plots/track_j_test_optimization.png")

    full_results = {
        "codebase_metrics": {
            "name": "Enterprise Distributed Workflow & DAG Orchestrator",
            "lines_of_code": 1150,
            "modules": ["dag.py", "task.py", "executor.py", "storage.py", "metrics.py", "test_suite.py"],
            "total_test_cases": len(test_cases)
        },
        "profiling": prof_data,
        "naive_baseline": res_naive,
        "target_optimization_180ms": {
            "budget_ms": target_budget,
            "ilp_solution": target_ilp,
            "speedup_ratio": target_speedup,
            "coverage_retention_pct": target_ilp["coverage_pct"]
        },
        "budget_scaling": {
            "budgets_ms": budgets,
            "ilp": ilp_results,
            "greedy": greedy_results
        }
    }

    with open("results/track_j_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print("[Track J] Completed successfully! Results written to results/track_j_results.json")
    return full_results


if __name__ == "__main__":
    run_swe_optimization_experiments()
