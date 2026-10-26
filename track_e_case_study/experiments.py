"""
Experiments for Track E: Main Case Study.
Smart Commercial Electric Vehicle Fleet Charging Optimization.
Compares Uncoordinated FCFS Baseline vs Smart Valley-Filling Heuristic vs Exact MILP.
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

from track_e_case_study.formulation import FleetChargingProblem
from track_e_case_study.baseline import UncoordinatedBaseline
from track_e_case_study.solver_milp import MILPFleetOptimizer
from track_e_case_study.solver_heuristic import SmartHeuristicOptimizer


def run_case_study_benchmarks():
    print("[Track E] Running EV Fleet Smart Charging Case Study Experiments...")
    problem = FleetChargingProblem(n_vehicles=20, transformer_limit_kw=250.0,
                                   demand_charge_per_kw=14.0, seed=42)

    baseline_solver = UncoordinatedBaseline()
    milp_solver = MILPFleetOptimizer()
    heuristic_solver = SmartHeuristicOptimizer()

    res_base = baseline_solver.solve(problem)
    res_milp = milp_solver.solve(problem)
    res_heur = heuristic_solver.solve(problem)

    cost_base = res_base["total_cost"]
    cost_milp = res_milp["total_cost"]
    cost_heur = res_heur["total_cost"]

    savings_milp_pct = ((cost_base - cost_milp) / cost_base) * 100.0
    savings_heur_pct = ((cost_base - cost_heur) / cost_base) * 100.0

    print(f"  Uncoordinated Baseline Cost:  ${cost_base:.2f} (Peak: {res_base['peak_demand_kw']:.1f} kW, Overload: {res_base['transformer_overload_hours']} hrs)")
    print(f"  Smart Heuristic Cost:         ${cost_heur:.2f} (Peak: {res_heur['peak_demand_kw']:.1f} kW, Savings: {savings_heur_pct:.1f}%)")
    print(f"  MILP (HiGHS) Optimal Cost:    ${cost_milp:.2f} (Peak: {res_milp['peak_demand_kw']:.1f} kW, Savings: {savings_milp_pct:.1f}%)")

    # 1. 24-Hour Load Profile & Tariff Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    hours = np.arange(24)
    ax1.plot(hours, res_base["hourly_load_kw"], label="Uncoordinated FCFS Baseline", color="crimson", linewidth=2.2, marker="o")
    ax1.plot(hours, res_heur["hourly_load_kw"], label="Smart Valley-Filling Heuristic", color="darkorange", linewidth=2.0, linestyle="--", marker="s")
    ax1.plot(hours, res_milp["hourly_load_kw"], label="MILP Optimal Schedule", color="forestgreen", linewidth=2.2, marker="^")
    ax1.axhline(problem.transformer_limit, color="darkred", linestyle=":", label=f"Transformer Limit ({problem.transformer_limit:.0f} kW)", linewidth=1.8)

    ax1.set_xlabel("Hour of Day (00:00 - 23:00)", fontsize=11)
    ax1.set_ylabel("Depot Grid Power Demand (kW)", fontsize=11, color="black")
    ax1.set_xticks(hours)
    ax1.grid(True, alpha=0.3)

    # Secondary axis for electricity tariff
    ax2 = ax1.twinx()
    ax2.step(hours, problem.tariffs, where="mid", color="royalblue", alpha=0.4, linewidth=2.0, label="TOU Electricity Tariff ($/kWh)")
    ax2.set_ylabel("Electricity Tariff ($/kWh)", fontsize=11, color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.title("Track E: 24-Hour EV Fleet Charging Load Shifting & Peak Shaving", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_e_load_profiles.png", dpi=300)
    plt.close()
    print("  Saved plots/track_e_load_profiles.png")

    # 2. Cost Breakdown Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Uncoordinated Baseline", "Smart Heuristic", "MILP Optimal"]
    energy_costs = [res_base["energy_cost"], res_heur["energy_cost"], res_milp["energy_cost"]]
    demand_penalties = [res_base["demand_penalty_cost"], res_heur["demand_penalty_cost"], res_milp["demand_penalty_cost"]]
    deg_costs = [res_base["degradation_cost"], res_heur["degradation_cost"], res_milp["degradation_cost"]]

    x_pos = np.arange(len(categories))
    width = 0.55

    p1 = ax.bar(x_pos, energy_costs, width, label="TOU Energy Cost ($)", color="#3498db")
    p2 = ax.bar(x_pos, demand_penalties, width, bottom=energy_costs, label="Peak Demand Charge ($)", color="#e74c3c")
    p3 = ax.bar(x_pos, deg_costs, width, bottom=np.array(energy_costs) + np.array(demand_penalties), label="Battery Degradation ($)", color="#2ecc71")

    ax.set_ylabel("Total Operational Cost ($)", fontsize=11)
    ax.set_title("Cost Breakdown Comparison for EV Commercial Fleet", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/track_e_cost_breakdown.png", dpi=300)
    plt.close()
    print("  Saved plots/track_e_cost_breakdown.png")

    results = {
        "problem_spec": {
            "num_vehicles": problem.n_vehicles,
            "transformer_limit_kw": problem.transformer_limit,
            "demand_charge_per_kw": problem.demand_charge,
            "degradation_cost_per_kwh": problem.deg_cost_per_kwh
        },
        "baseline": res_base,
        "heuristic": res_heur,
        "milp_optimal": res_milp,
        "comparison": {
            "milp_cost_savings_pct": savings_milp_pct,
            "heuristic_cost_savings_pct": savings_heur_pct,
            "peak_reduction_kw": res_base["peak_demand_kw"] - res_milp["peak_demand_kw"],
            "optimality_gap_heuristic_vs_milp_pct": ((cost_heur - cost_milp) / cost_milp) * 100.0
        }
    }

    with open("results/track_e_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[Track E] Completed successfully! Results written to results/track_e_results.json")
    return results


if __name__ == "__main__":
    run_case_study_benchmarks()
