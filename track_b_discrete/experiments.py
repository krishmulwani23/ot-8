"""
Experiments for Track B: Discrete Optimization.
Benchmarks:
1. Exact vs Heuristic on 0-1 Knapsack (Greedy, DP, Branch-and-Bound, ILP)
2. Scalability across problem dimensions N in [10, 20, 50, 100, 200]
3. QUBO & Ising formulation of Max-Cut and classical solution as baseline for Track C.
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

from track_b_discrete.solvers import (
    KnapsackInstance, GreedyKnapsackSolver, DynamicProgrammingKnapsackSolver,
    BranchAndBoundKnapsackSolver, ILPKnapsackSolver
)
from track_b_discrete.qubo_ising import (
    QUBOInstance, MaxCutProblem, ClassicalQUBOSolvers
)


def run_discrete_comparison():
    print("[Track B] Running Knapsack Solver Comparisons across problem sizes...")
    rng = np.random.default_rng(42)
    sizes = [10, 20, 30, 50, 100]
    results_by_size = {}

    greedy_solver = GreedyKnapsackSolver()
    dp_solver = DynamicProgrammingKnapsackSolver()
    bb_solver = BranchAndBoundKnapsackSolver()
    ilp_solver = ILPKnapsackSolver()

    runtimes = {"Greedy": [], "DP": [], "BranchAndBound": [], "ILP_HiGHS": []}
    values = {"Greedy": [], "DP": [], "BranchAndBound": [], "ILP_HiGHS": []}

    for n in sizes:
        # Generate reproducible benchmark instance
        weights = rng.integers(1, 50, size=n).astype(float)
        vals = weights * rng.uniform(1.0, 3.0, size=n) + rng.uniform(5, 20, size=n)
        capacity = float(np.sum(weights) * 0.45) # 45% capacity ratio
        inst = KnapsackInstance(vals, weights, capacity)

        g_res = greedy_solver.solve(inst)
        ilp_res = ilp_solver.solve(inst)

        # For large N, DP with very large capacity or B&B worst case can be slow
        # Cap DP capacity to avoid memory blowout
        dp_res = dp_solver.solve(inst) if n <= 100 else None
        bb_res = bb_solver.solve(inst) if n <= 30 else None

        results_by_size[str(n)] = {
            "Greedy": g_res,
            "ILP_HiGHS": ilp_res,
            "DP": dp_res,
            "BranchAndBound": bb_res
        }

        runtimes["Greedy"].append(g_res["runtime_ms"])
        runtimes["ILP_HiGHS"].append(ilp_res["runtime_ms"])
        values["Greedy"].append(g_res["total_value"])
        values["ILP_HiGHS"].append(ilp_res["total_value"])

        if dp_res:
            runtimes["DP"].append(dp_res["runtime_ms"])
            values["DP"].append(dp_res["total_value"])
        else:
            runtimes["DP"].append(None)
            values["DP"].append(None)

        if bb_res:
            runtimes["BranchAndBound"].append(bb_res["runtime_ms"])
            values["BranchAndBound"].append(bb_res["total_value"])
        else:
            runtimes["BranchAndBound"].append(None)
            values["BranchAndBound"].append(None)

    # Plot runtime scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, runtimes["Greedy"], marker="o", label="Greedy Heuristic", linewidth=2)
    ax.plot(sizes, runtimes["ILP_HiGHS"], marker="s", label="ILP (HiGHS Branch-and-Cut)", linewidth=2)
    valid_dp = [(s, r) for s, r in zip(sizes, runtimes["DP"]) if r is not None]
    if valid_dp:
        ax.plot([v[0] for v in valid_dp], [v[1] for v in valid_dp], marker="^", label="Dynamic Programming", linewidth=2)
    valid_bb = [(s, r) for s, r in zip(sizes, runtimes["BranchAndBound"]) if r is not None]
    if valid_bb:
        ax.plot([v[0] for v in valid_bb], [v[1] for v in valid_bb], marker="d", label="Branch-and-Bound (Custom)", linewidth=2)

    ax.set_yscale("log")
    ax.set_title("Discrete Solvers Runtime Scaling with Problem Size N", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Items (N)", fontsize=10)
    ax.set_ylabel("Runtime (ms) [Log Scale]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_b_scaling.png", dpi=300)
    plt.close()
    print("  Saved plots/track_b_scaling.png")
    return results_by_size


def run_qubo_baseline_for_quantum():
    print("[Track B] Formulating Max-Cut QUBO/Ising & Classical Benchmark Baseline for Track C...")
    # Canonical 6-node Regular / Ring Graph with cross edges for Max-Cut
    # Vertex count N=6 is perfect for direct classical brute-force, SA, and exact quantum simulation
    edges = [
        (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 0, 1.0), # ring
        (0, 3, 1.0), (1, 4, 1.0) # cross chords
    ]
    maxcut = MaxCutProblem(n_vertices=6, edges=edges)
    qubo = maxcut.to_qubo()
    J, h, offset = qubo.to_ising()

    print(f"  Max-Cut Graph: {maxcut.n} vertices, {len(edges)} edges")
    print(f"  QUBO Matrix shape: {qubo.Q.shape}")
    print(f"  Ising Hamiltonian: {len(h)} local fields, offset={offset:.2f}")

    # Classical Solvers
    exact_res = ClassicalQUBOSolvers.brute_force(qubo)
    sa_res = ClassicalQUBOSolvers.simulated_annealing(qubo, n_steps=2000, seed=42)

    best_x_exact = np.array(exact_res["best_x"])
    best_cut_exact = maxcut.calculate_cut(best_x_exact)

    best_x_sa = np.array(sa_res["best_x"])
    best_cut_sa = maxcut.calculate_cut(best_x_sa)

    print(f"  Exact Classical Optimal Cut: {best_cut_exact} (Energy: {exact_res['best_energy']:.2f})")
    print(f"  Simulated Annealing Cut:     {best_cut_sa} (Energy: {sa_res['best_energy']:.2f})")

    qubo_baseline = {
        "num_vertices": 6,
        "edges": edges,
        "exact_optimal_cut": best_cut_exact,
        "exact_best_x": exact_res["best_x"],
        "exact_energy": exact_res["best_energy"],
        "exact_runtime_ms": exact_res["runtime_ms"],
        "sa_cut": best_cut_sa,
        "sa_best_x": sa_res["best_x"],
        "sa_energy": sa_res["best_energy"],
        "sa_runtime_ms": sa_res["runtime_ms"],
        "qubo_matrix": qubo.Q.tolist(),
        "ising_couplings_J": J.tolist(),
        "ising_fields_h": h.tolist(),
        "ising_offset": offset
    }

    with open("results/track_b_qubo_baseline.json", "w") as f:
        json.dump(qubo_baseline, f, indent=2)
    print("  Saved results/track_b_qubo_baseline.json")
    return qubo_baseline


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    discrete_res = run_discrete_comparison()
    qubo_baseline = run_qubo_baseline_for_quantum()

    full_results = {
        "knapsack_benchmark": discrete_res,
        "qubo_baseline_for_quantum": qubo_baseline
    }

    with open("results/track_b_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print("[Track B] Completed successfully! Results written to results/track_b_results.json")
    return full_results


if __name__ == "__main__":
    main()
