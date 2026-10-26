"""
Experiments for Track F: Compiler Optimization.
Benchmarks:
1. Optimization passes: Constant Folding, Dead Code Elimination, CSE, Strength Reduction
2. Semantic Equivalence Verification: Output(Unoptimized) == Output(Optimized)
3. Instruction count, cycle count, code size, and compile time profiling
4. ILP Graph-Coloring Register Allocation vs Register Pool Size K
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track_f_compiler.ir import Instruction
from track_f_compiler.optimizations import OptimizationPipeline
from track_f_compiler.register_allocator import InterferenceGraph, ILPRegisterAllocator
from track_f_compiler.virtual_machine import TACVirtualMachine


def create_benchmark_program_1() -> List[Instruction]:
    """
    Benchmark 1: Arithmetic Redundancy & Dead Code.
    Contains foldable constants, repeated subexpressions (a + b), dead computations.
    """
    return [
        Instruction('assign', dest='a', src1='10'),
        Instruction('assign', dest='b', src1='20'),
        Instruction('add', dest='c', src1='a', src2='b'),        # c = 10 + 20 = 30
        Instruction('mul', dest='d', src1='c', src2='2'),         # d = c * 2 -> strength reduction
        Instruction('add', dest='dead1', src1='a', src2='100'),    # dead code (never read)
        Instruction('mul', dest='dead2', src1='dead1', src2='2'),  # dead code
        Instruction('add', dest='e', src1='a', src2='b'),        # CSE: identical to c = a + b!
        Instruction('add', dest='total', src1='d', src2='e'),     # total = d + e
        Instruction('print', src1='total'),
        Instruction('ret', src1='total')
    ]


def create_benchmark_program_2() -> List[Instruction]:
    """
    Benchmark 2: Loop Accumulation with Invariants and Strength Reduction.
    Loop from i = 0 to 100: sum += (i * 2) + invariant
    """
    return [
        Instruction('assign', dest='sum', src1='0'),
        Instruction('assign', dest='i', src1='0'),
        Instruction('assign', dest='n', src1='50'),
        Instruction('assign', dest='inv1', src1='5'),
        Instruction('assign', dest='inv2', src1='7'),
        Instruction('add', dest='invariant', src1='inv1', src2='inv2'), # invariant = 12 (foldable)
        Instruction('label', dest='loop_start'),
        Instruction('lt', dest='cond', src1='i', src2='n'),
        Instruction('jmp_false', dest='loop_end', src1='cond'),
        Instruction('mul', dest='term', src1='i', src2='2'),            # strength reduction to i + i
        Instruction('add', dest='val', src1='term', src2='invariant'),
        Instruction('add', dest='sum', src1='sum', src2='val'),
        Instruction('add', dest='i', src1='i', src2='1'),
        Instruction('jmp', dest='loop_start'),
        Instruction('label', dest='loop_end'),
        Instruction('print', src1='sum'),
        Instruction('ret', src1='sum')
    ]


def run_compiler_experiments():
    print("[Track F] Running Compiler Optimization Passes & Profiling...")
    vm = TACVirtualMachine()

    benchmarks = [
        ("Prog1_Arithmetic_CSE_DCE", create_benchmark_program_1()),
        ("Prog2_Loop_Accumulation", create_benchmark_program_2())
    ]

    bench_results = {}

    for name, prog_unopt in benchmarks:
        # Measure unoptimized
        t0 = time.perf_counter()
        res_unopt = vm.execute(prog_unopt)
        t_unopt_ms = (time.perf_counter() - t0) * 1000.0

        # Run optimization passes
        t0 = time.perf_counter()
        prog_opt = OptimizationPipeline.run_all_passes(prog_unopt)
        compile_time_ms = (time.perf_counter() - t0) * 1000.0

        # Measure optimized
        t0 = time.perf_counter()
        res_opt = vm.execute(prog_opt)
        t_opt_ms = (time.perf_counter() - t0) * 1000.0

        # STRICT SEMANTIC EQUIVALENCE CHECK
        assert res_unopt["output"] == res_opt["output"], \
            f"Semantic mismatch in {name}! Unopt: {res_unopt['output']}, Opt: {res_opt['output']}"

        inst_reduction = len(prog_unopt) - len(prog_opt)
        cycle_reduction = res_unopt["cycles"] - res_opt["cycles"]
        cycle_speedup = res_unopt["cycles"] / max(1, res_opt["cycles"])

        print(f"  {name}:")
        print(f"    Correctness Check: PASS (Output: {res_opt['output']})")
        print(f"    Code Size (Instructions): {len(prog_unopt)} -> {len(prog_opt)} (-{inst_reduction})")
        print(f"    Executed Cycles:          {res_unopt['cycles']} -> {res_opt['cycles']} (Speedup: {cycle_speedup:.2f}x)")
        print(f"    Compile Optimization Time: {compile_time_ms:.3f} ms")

        bench_results[name] = {
            "output_unopt": res_unopt["output"],
            "output_opt": res_opt["output"],
            "correctness_verified": True,
            "unopt_instructions": len(prog_unopt),
            "opt_instructions": len(prog_opt),
            "instruction_reduction": inst_reduction,
            "unopt_cycles": res_unopt["cycles"],
            "opt_cycles": res_opt["cycles"],
            "cycle_speedup": cycle_speedup,
            "compile_time_ms": compile_time_ms
        }

    # Plot Instruction and Cycle Reduction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = list(bench_results.keys())
    x = np.arange(len(names))
    w = 0.35

    unopt_insts = [bench_results[n]["unopt_instructions"] for n in names]
    opt_insts = [bench_results[n]["opt_instructions"] for n in names]
    ax1.bar(x - w/2, unopt_insts, w, label="Unoptimized TAC", color="#e74c3c")
    ax1.bar(x + w/2, opt_insts, w, label="Optimized TAC", color="#2ecc71")
    ax1.set_title("Static Instruction Count (Code Size)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Instructions")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Prog 1 (Arith/CSE)", "Prog 2 (Loop/Strength)"], fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)

    unopt_cycles = [bench_results[n]["unopt_cycles"] for n in names]
    opt_cycles = [bench_results[n]["opt_cycles"] for n in names]
    ax2.bar(x - w/2, unopt_cycles, w, label="Unoptimized Cycles", color="#e74c3c")
    ax2.bar(x + w/2, opt_cycles, w, label="Optimized Cycles", color="#2ecc71")
    ax2.set_title("Dynamic Executed Cycles (Runtime)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("VM Clock Cycles")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Prog 1 (Arith/CSE)", "Prog 2 (Loop/Strength)"], fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_f_compiler_gains.png", dpi=300)
    plt.close()
    print("  Saved plots/track_f_compiler_gains.png")

    # 2. Register Allocation via ILP
    print("[Track F] Solving Register Allocation as ILP Optimization Problem...")
    prog1 = create_benchmark_program_1()
    ig = InterferenceGraph(prog1)

    print(f"  Variables to allocate: {len(ig.variables)} ({ig.variables})")
    print(f"  Interference edges:    {len(ig.interferences)}")

    k_values = [2, 3, 4, 5]
    reg_alloc_results = {}
    spill_counts = []

    for k in k_values:
        allocator = ILPRegisterAllocator(k_registers=k)
        res_alloc = allocator.allocate(ig)
        reg_alloc_results[f"K={k}"] = res_alloc
        spill_counts.append(res_alloc["spill_count"])
        print(f"    K={k} Registers -> Spills: {res_alloc['spill_count']} (Allocated: {res_alloc['allocation']})")

    # Plot spills vs K
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, spill_counts, marker="o", color="purple", linewidth=2.2)
    ax.set_title("ILP Register Allocation: Spills vs Available Register Pool (K)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Physical Registers (K)", fontsize=10)
    ax.set_ylabel("Spilled Variables to Memory", fontsize=10)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/track_f_register_allocation.png", dpi=300)
    plt.close()
    print("  Saved plots/track_f_register_allocation.png")

    full_results = {
        "benchmarks": bench_results,
        "register_allocation_ilp": reg_alloc_results
    }

    with open("results/track_f_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print("[Track F] Completed successfully! Results written to results/track_f_results.json")
    return full_results


if __name__ == "__main__":
    run_compiler_experiments()
