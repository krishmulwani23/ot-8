"""
Experiments for Track H: DBMS Optimization.
Benchmarks:
1. Multi-Index Selection formulated as Knapsack / ILP under storage budget
2. Verification of Query Plans (EXPLAIN QUERY PLAN) and Result Equivalence
3. Workload Latency and Throughput Before vs After Optimization
4. Buffer Pool Cache Replacement Benchmark (LRU vs 2Q vs Clock)
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

from track_h_dbms.database import TPCHDatabase
from track_h_dbms.index_selection import IndexSelectionOptimizer
from track_h_dbms.buffer_cache import benchmark_buffer_pool


def run_dbms_experiments():
    print("[Track H] Initializing TPC-H Database and Workload...")
    db = TPCHDatabase(seed=42)
    queries = db.get_queries()

    print(f"  Populated database tables successfully.")
    print(f"  Testing queries: {list(queries.keys())}")

    # 1. Capture Baseline Latencies and EXPLAIN plans (WITHOUT indexes)
    cur = db.conn.cursor()
    base_plans = {}
    base_results = {}
    base_latencies = {}

    for qname, qsql in queries.items():
        cur.execute(f"EXPLAIN QUERY PLAN {qsql}")
        plan_rows = cur.fetchall()
        base_plans[qname] = [f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in plan_rows]

        t0 = time.perf_counter()
        cur.execute(qsql)
        res = cur.fetchall()
        t_ms = (time.perf_counter() - t0) * 1000.0

        base_results[qname] = [tuple(r) for r in res]
        base_latencies[qname] = t_ms

    print("  Baseline Unindexed Latencies:")
    for qname, lat in base_latencies.items():
        print(f"    {qname}: {lat:.2f} ms")

    # 2. Formulate & Solve Index Selection Problem
    print("  Evaluating Index Benefits and Solving Storage-Constrained Knapsack...")
    opt = IndexSelectionOptimizer(db.conn)
    benefits = opt.evaluate_benefits(queries)

    for i, cand in enumerate(opt.candidates):
        print(f"    Index {cand.name} ({cand.size_kb} KB): Benefit = {benefits[i]:.2f} ms")

    budget_kb = 400 # storage limit 400 KB
    greedy_sol = opt.solve_greedy(benefits, budget_kb)
    dp_sol = opt.solve_dynamic_programming(benefits, budget_kb)
    ilp_sol = opt.solve_ilp(benefits, budget_kb)

    print(f"  Greedy Selection:     {greedy_sol['selected_indexes']} (Benefit: {greedy_sol['total_benefit_ms']:.1f}ms, Size: {greedy_sol['total_size_kb']}KB)")
    print(f"  DP Knapsack Solution: {dp_sol['selected_indexes']} (Benefit: {dp_sol['total_benefit_ms']:.1f}ms, Size: {dp_sol['total_size_kb']}KB)")
    print(f"  ILP (HiGHS) Solution: {ilp_sol['selected_indexes']} (Benefit: {ilp_sol['total_benefit_ms']:.1f}ms, Size: {ilp_sol['total_size_kb']}KB)")

    # 3. Apply Selected Optimal Indexes to Database
    selected_idx_names = ilp_sol["selected_indexes"]
    for cand in opt.candidates:
        if cand.name in selected_idx_names:
            cur.execute(cand.create_sql)
    db.conn.commit()

    # 4. Measure Optimized Latencies and Verify Semantic Result Equivalence
    opt_plans = {}
    opt_latencies = {}

    for qname, qsql in queries.items():
        cur.execute(f"EXPLAIN QUERY PLAN {qsql}")
        plan_rows = cur.fetchall()
        opt_plans[qname] = [f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in plan_rows]

        t0 = time.perf_counter()
        cur.execute(qsql)
        res = cur.fetchall()
        t_ms = (time.perf_counter() - t0) * 1000.0

        opt_res = [tuple(r) for r in res]
        opt_latencies[qname] = t_ms

        # STRICT RESULT IDENTITY VERIFICATION
        assert base_results[qname] == opt_res, f"Query result mismatch in {qname} after indexing!"

    print("  Optimized Latencies (with Selected Indexes):")
    total_base_t = sum(base_latencies.values())
    total_opt_t = sum(opt_latencies.values())
    for qname, lat in opt_latencies.items():
        speedup = base_latencies[qname] / max(0.01, lat)
        print(f"    {qname}: {lat:.2f} ms (Speedup: {speedup:.2f}x)")

    overall_speedup = total_base_t / max(0.01, total_opt_t)
    print(f"  Overall Workload Speedup: {overall_speedup:.2f}x ({total_base_t:.1f}ms -> {total_opt_t:.1f}ms)")
    print(f"  Correctness Verified: All query results 100% BIT-FOR-BIT IDENTICAL!")

    # 5. Plot Query Latency Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    q_labels = ["Q1 (Aggr)", "Q2 (Cust-Orders)", "Q3 (3-Way Join)"]
    x = np.arange(len(q_labels))
    w = 0.35

    ax.bar(x - w/2, [base_latencies[q] for q in queries.keys()], w, label="Baseline (No Index / Full Scan)", color="#e74c3c")
    ax.bar(x + w/2, [opt_latencies[q] for q in queries.keys()], w, label="Optimized (Knapsack Selected Indexes)", color="#2ecc71")
    ax.set_title("DBMS Query Execution Latency Before vs After Index Optimization", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms) [Lower is Better]", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_h_dbms_latency.png", dpi=300)
    plt.close()
    print("  Saved plots/track_h_dbms_latency.png")

    # 6. Buffer Pool Simulation
    print("  Benchmarking Buffer Pool Replacement Policies (LRU vs 2Q vs Clock)...")
    rng = np.random.default_rng(123)
    # Generate realistic access trace: Zipfian hot pages + periodic sequential scan
    hot_pages = rng.integers(1, 40, size=1500)
    scan_pages = np.arange(100, 350) # sequential scan burst
    mixed_trace = np.concatenate([hot_pages[:700], scan_pages, hot_pages[700:]]).tolist()

    capacities = [20, 40, 80, 120, 200]
    bcache_res = benchmark_buffer_pool(mixed_trace, capacities)

    # Plot buffer hit ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(capacities, bcache_res["LRU"], marker="o", label="LRU (Standard)", linewidth=2)
    ax.plot(capacities, bcache_res["TwoQueue"], marker="s", label="2Q (Scan Resistant)", linewidth=2)
    ax.plot(capacities, bcache_res["Clock"], marker="^", label="Clock (Second Chance)", linewidth=2, linestyle="--")
    ax.set_title("Buffer Pool Hit Ratio: Scan Resistance Comparison", fontsize=12, fontweight="bold")
    ax.set_xlabel("Buffer Pool Size (Pages)", fontsize=10)
    ax.set_ylabel("Hit Ratio (Hits / Total References)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/track_h_buffer_cache.png", dpi=300)
    plt.close()
    print("  Saved plots/track_h_buffer_cache.png")

    full_results = {
        "index_selection": {
            "budget_kb": budget_kb,
            "greedy": greedy_sol,
            "dynamic_programming": dp_sol,
            "ilp_highs": ilp_sol
        },
        "query_performance": {
            "baseline_latencies_ms": base_latencies,
            "optimized_latencies_ms": opt_latencies,
            "overall_speedup": overall_speedup,
            "total_baseline_ms": total_base_t,
            "total_optimized_ms": total_opt_t,
            "results_identical": True
        },
        "query_plans_explain": {
            "baseline_plans": base_plans,
            "optimized_plans": opt_plans
        },
        "buffer_cache_policies": bcache_res
    }

    with open("results/track_h_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print("[Track H] Completed successfully! Results written to results/track_h_results.json")
    return full_results


if __name__ == "__main__":
    run_dbms_experiments()
