"""
Experiments for Track G: xv6 Operating-System Optimization.
Runs identical workloads under Default Round-Robin vs MLFQ Scheduler,
benchmarks Response Time, Turnaround Time, Context Switches, and Fairness,
and generates Gantt execution timelines.
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

from track_g_xv6.xv6_kernel_sim import SimProcess, XV6KernelSimulator

def create_benchmark_workload():
    """Constructs the 3 concurrent benchmark workloads for fair comparison."""
    # Workload 1: CPU-bound (Long computation, 2 bursts of 35 ticks each, separated by 1 brief IO)
    p_cpu = SimProcess(pid=1, name="cpu_bound", arrival_tick=0,
                       cpu_bursts=[35, 35], io_bursts=[2])

    # Workload 2: I/O-bound (12 bursts of 2 CPU ticks, 4 IO ticks)
    p_io = SimProcess(pid=2, name="io_bound", arrival_tick=0,
                      cpu_bursts=[2] * 12, io_bursts=[4] * 11)

    # Workload 3: Mixed Interactive (6 bursts of 6 CPU ticks, 6 IO ticks)
    p_interactive = SimProcess(pid=3, name="mixed_interactive", arrival_tick=2,
                               cpu_bursts=[6] * 6, io_bursts=[6] * 5)

    return [p_cpu, p_io, p_interactive]


def run_xv6_experiments():
    print("[Track G] Benchmarking xv6 Default Round-Robin vs MLFQ Scheduler...")

    # 1. Run Round-Robin Baseline
    procs_rr = create_benchmark_workload()
    sim_rr = XV6KernelSimulator(scheduler_type="round_robin")
    res_rr = sim_rr.run(procs_rr, max_ticks=400)

    # 2. Run MLFQ Scheduler
    procs_mlfq = create_benchmark_workload()
    sim_mlfq = XV6KernelSimulator(scheduler_type="mlfq", boost_interval=100)
    res_mlfq = sim_mlfq.run(procs_mlfq, max_ticks=400)

    print("  Round-Robin Baseline:")
    print(f"    Mean Turnaround Time: {res_rr['mean_turnaround']:.1f} ticks")
    print(f"    Mean Response Time:   {res_rr['mean_response']:.1f} ticks")
    print(f"    Context Switches:     {res_rr['total_context_switches']}")
    print("  MLFQ Optimized Scheduler:")
    print(f"    Mean Turnaround Time: {res_mlfq['mean_turnaround']:.1f} ticks")
    print(f"    Mean Response Time:   {res_mlfq['mean_response']:.1f} ticks")
    print(f"    Context Switches:     {res_mlfq['total_context_switches']}")

    # Specific response time speedup for interactive and I/O processes
    io_resp_rr = res_rr["process_stats"]["io_bound"]["response_time"]
    io_resp_mlfq = res_mlfq["process_stats"]["io_bound"]["response_time"]
    inter_resp_rr = res_rr["process_stats"]["mixed_interactive"]["response_time"]
    inter_resp_mlfq = res_mlfq["process_stats"]["mixed_interactive"]["response_time"]

    print(f"    I/O Process Response Time:       {io_resp_rr} -> {io_resp_mlfq} ticks")
    print(f"    Interactive Proc Response Time:  {inter_resp_rr} -> {inter_resp_mlfq} ticks")

    # Plot Comparison Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    procs = ["cpu_bound", "io_bound", "mixed_interactive"]
    x = np.arange(len(procs))
    w = 0.35

    # Response Time Plot
    resp_rr_vals = [res_rr["process_stats"][p]["response_time"] for p in procs]
    resp_mlfq_vals = [res_mlfq["process_stats"][p]["response_time"] for p in procs]
    ax1.bar(x - w/2, resp_rr_vals, w, label="Default Round-Robin", color="#e74c3c")
    ax1.bar(x + w/2, resp_mlfq_vals, w, label="MLFQ Scheduler", color="#2ecc71")
    ax1.set_title("Response Time by Workload (Ticks to First CPU Schedule)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Ticks (lower is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["CPU-Bound", "I/O-Bound", "Interactive"], fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    # Turnaround Time Plot
    turn_rr_vals = [res_rr["process_stats"][p]["turnaround_time"] for p in procs]
    turn_mlfq_vals = [res_mlfq["process_stats"][p]["turnaround_time"] for p in procs]
    ax2.bar(x - w/2, turn_rr_vals, w, label="Default Round-Robin", color="#e74c3c")
    ax2.bar(x + w/2, turn_mlfq_vals, w, label="MLFQ Scheduler", color="#2ecc71")
    ax2.set_title("Turnaround Time by Workload (Ticks to Completion)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Ticks (lower is better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["CPU-Bound", "I/O-Bound", "Interactive"], fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_g_xv6_scheduler.png", dpi=300)
    plt.close()
    print("  Saved plots/track_g_xv6_scheduler.png")

    # Save Gantt log slice for visual timeline
    fig, (g_ax1, g_ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    max_plot_ticks = 80

    for res, ax, title in [(res_rr, g_ax1, "xv6 Default Round-Robin Scheduler"),
                           (res_mlfq, g_ax2, "xv6 MLFQ Scheduler with Dynamic Priority Queues")]:
        pid_map = {"cpu_bound": 1, "io_bound": 2, "mixed_interactive": 3, "IDLE": 0}
        colors = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71", 0: "#bdc3c7"}
        log_slice = res["gantt_log"][:max_plot_ticks]

        for entry in log_slice:
            t = entry["tick"]
            name = entry["name"]
            p_id = pid_map.get(name, 0)
            ax.barh(p_id, 1, left=t-1, color=colors.get(p_id, "gray"), edgecolor="black", linewidth=0.3)

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["IDLE", "CPU", "I/O", "Interactive"], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

    g_ax2.set_xlabel("Time (Timer Ticks)", fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/track_g_xv6_gantt.png", dpi=300)
    plt.close()
    print("  Saved plots/track_g_xv6_gantt.png")

    results = {
        "round_robin_baseline": {
            "mean_turnaround": res_rr["mean_turnaround"],
            "mean_response": res_rr["mean_response"],
            "total_context_switches": res_rr["total_context_switches"],
            "jains_fairness": res_rr["jains_fairness"],
            "process_stats": res_rr["process_stats"]
        },
        "mlfq_optimized": {
            "mean_turnaround": res_mlfq["mean_turnaround"],
            "mean_response": res_mlfq["mean_response"],
            "total_context_switches": res_mlfq["total_context_switches"],
            "jains_fairness": res_mlfq["jains_fairness"],
            "process_stats": res_mlfq["process_stats"]
        },
        "improvements": {
            "mean_turnaround_improvement_pct": ((res_rr["mean_turnaround"] - res_mlfq["mean_turnaround"]) / res_rr["mean_turnaround"]) * 100.0,
            "mean_response_improvement_pct": ((res_rr["mean_response"] - res_mlfq["mean_response"]) / max(1e-6, res_rr["mean_response"])) * 100.0,
            "io_response_speedup": (io_resp_rr / max(1, io_resp_mlfq)),
            "interactive_response_speedup": (inter_resp_rr / max(1, inter_resp_mlfq))
        }
    }

    with open("results/track_g_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[Track G] Completed successfully! Results written to results/track_g_results.json")
    return results


if __name__ == "__main__":
    run_xv6_experiments()
