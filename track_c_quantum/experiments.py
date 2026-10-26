"""
Experiments for Track C: Quantum Optimization.
Connects directly to Track B Max-Cut QUBO baseline, benchmarks QAOA,
evaluates Parameter-Shift vs SPSA, finite measurement shots, and barren plateaus.
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

from track_c_quantum.qaoa_vqe import QAOAOptimizer, VQEOptimizer
from track_c_quantum.noise_and_barren_plateaus import QuantumNoiseAndScaling


def run_qaoa_vs_classical_experiment():
    print("[Track C] Running QAOA on Track B Max-Cut Graph Instance...")
    # Load Track B baseline if exists, else canonical 6-vertex graph
    track_b_path = "results/track_b_qubo_baseline.json"
    if os.path.exists(track_b_path):
        with open(track_b_path, "r") as f:
            b_data = json.load(f)
        edges = [(e[0], e[1], float(e[2])) for e in b_data["edges"]]
        n_qubits = b_data["num_vertices"]
        classical_opt_cut = b_data["exact_optimal_cut"]
    else:
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 0, 1.0), (0, 3, 1.0), (1, 4, 1.0)]
        n_qubits = 6
        classical_opt_cut = 8.0

    qaoa = QAOAOptimizer(n_qubits=n_qubits, edges=edges, p_layers=2)

    t0 = time.perf_counter()
    res_ps = qaoa.optimize_classical_loop(max_iter=30, lr=0.1, method="parameter_shift")
    t_ps = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    res_spsa = qaoa.optimize_classical_loop(max_iter=30, lr=0.1, method="spsa")
    t_spsa = (time.perf_counter() - t0) * 1000.0

    print(f"  Classical Optimal Cut (Track B): {classical_opt_cut:.1f}")
    print(f"  QAOA (Parameter-Shift) Cut:    {res_ps['sampled_cut']:.1f} (Runtime: {t_ps:.1f}ms, Best State: {res_ps['best_bitstring']})")
    print(f"  QAOA (SPSA) Cut:               {res_spsa['sampled_cut']:.1f} (Runtime: {t_spsa:.1f}ms, Best State: {res_spsa['best_bitstring']})")

    # Plot QAOA cut convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res_ps["history_cuts"], label="QAOA (Parameter-Shift Rule)", linewidth=2)
    ax.plot(res_spsa["history_cuts"], label="QAOA (SPSA - 2 evals/step)", linewidth=2, linestyle="--")
    ax.axhline(classical_opt_cut, color="red", linestyle=":", label=f"Classical Optimum Cut ({classical_opt_cut})")
    ax.set_title("QAOA Max-Cut Optimization vs Classical Baseline (N=6 Qubits, p=2)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=10)
    ax.set_ylabel("Expected Cut Value", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_c_qaoa_convergence.png", dpi=300)
    plt.close()
    print("  Saved plots/track_c_qaoa_convergence.png")

    return {
        "classical_opt_cut": classical_opt_cut,
        "qaoa_parameter_shift": {
            "sampled_cut": res_ps["sampled_cut"],
            "expected_cut": res_ps["best_cut_expected"],
            "best_bitstring": res_ps["best_bitstring"],
            "runtime_ms": t_ps,
            "approximation_ratio": res_ps["sampled_cut"] / classical_opt_cut
        },
        "qaoa_spsa": {
            "sampled_cut": res_spsa["sampled_cut"],
            "expected_cut": res_spsa["best_cut_expected"],
            "best_bitstring": res_spsa["best_bitstring"],
            "runtime_ms": t_spsa,
            "approximation_ratio": res_spsa["sampled_cut"] / classical_opt_cut
        }
    }


def run_finite_shots_and_noise_experiment():
    print("[Track C] Running Finite Measurement Shots & Noise Analysis...")
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]
    qaoa = QAOAOptimizer(n_qubits=3, edges=edges, p_layers=1)
    qc = qaoa.build_circuit(gamma=np.array([0.3]), beta=np.array([0.4]))

    shot_counts = [50, 100, 250, 500, 1000, 2500, 5000]
    shot_results = QuantumNoiseAndScaling.finite_shot_analysis(qc, qaoa.diag_cost, shot_counts=shot_counts, trials=25)

    stds = [shot_results[str(s)]["std_deviation"] for s in shot_counts]
    theory_stds = [1.0 / np.sqrt(s) for s in shot_counts]

    # Plot shot noise scaling
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shot_counts, stds, marker="o", label="Empirical Standard Deviation", linewidth=2)
    ax.plot(shot_counts, theory_stds, marker="s", label=r"Theoretical Shot Noise $\sim 1/\sqrt{N_{\rm shots}}$", linestyle="--", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Measurement Shot Noise vs Number of Measurement Shots", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Measurement Shots", fontsize=10)
    ax.set_ylabel("Standard Deviation", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/track_c_shot_noise.png", dpi=300)
    plt.close()
    print("  Saved plots/track_c_shot_noise.png")

    return shot_results


def run_barren_plateaus_experiment():
    print("[Track C] Investigating Barren Plateaus (Gradient Variance Decay with Qubit Count)...")
    qubits = [2, 4, 6, 8]
    bp_data = QuantumNoiseAndScaling.study_barren_plateaus(qubit_counts=qubits, depth=2, n_samples=30)

    # Plot variance vs qubits
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qubits, bp_data["gradient_variances"], marker="o", color="crimson", linewidth=2, label="Measured Var[grad]")
    # Fit theoretical 2^{-n} line
    ref_line = [bp_data["gradient_variances"][0] * (2.0 ** -(q - qubits[0])) for q in qubits]
    ax.plot(qubits, ref_line, linestyle="--", color="gray", label=r"Exponential Decay $\sim O(2^{-n})$")
    ax.set_yscale("log")
    ax.set_title("Barren Plateau Phenomenon: Gradient Variance vs Qubit Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Qubits (n)", fontsize=10)
    ax.set_ylabel("Variance of Parameter Gradient [Log Scale]", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/track_c_barren_plateaus.png", dpi=300)
    plt.close()
    print("  Saved plots/track_c_barren_plateaus.png")

    return bp_data


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    qaoa_res = run_qaoa_vs_classical_experiment()
    shot_res = run_finite_shots_and_noise_experiment()
    bp_res = run_barren_plateaus_experiment()

    # Quick VQE demonstration
    vqe = VQEOptimizer(n_qubits=3, layers=1)
    diag_h = np.array([2.5, 1.0, -1.5, 0.5, 3.0, -2.0, -0.5, 1.2]) # Ground state = -2.0
    vqe_res = vqe.ground_state_search(diag_h, max_iter=25, lr=0.15)
    print(f"  VQE Ground State Energy: {vqe_res['final_energy']:.3f} (Exact: {vqe_res['exact_ground_energy']:.3f})")

    results = {
        "qaoa_vs_classical": qaoa_res,
        "shot_noise_analysis": shot_res,
        "barren_plateaus": bp_res,
        "vqe_summary": {
            "final_energy": vqe_res["final_energy"],
            "exact_ground_energy": vqe_res["exact_ground_energy"],
            "energy_error": vqe_res["energy_error"]
        }
    }

    with open("results/track_c_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[Track C] Completed successfully! Results written to results/track_c_results.json")
    return results


if __name__ == "__main__":
    main()
