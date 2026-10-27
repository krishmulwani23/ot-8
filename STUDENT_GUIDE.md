# STUDENT WORKFLOW & VIVA DEFENSE GUIDE
## Assignment 8: "Optimize Anything — The 14-Day Vibe-Coding Challenge"
**Course**: Optimization Techniques  
**Repository**: `c:\Users\HP\Documents\ot-8`

---

## 1. What Steps You Should Do Next (Action Checklist)

Follow these exact steps to verify, run, push to GitHub, and defend your submission:

### Step 1: Activate the Python Environment
Open PowerShell or your terminal in this repository directory (`c:\Users\HP\Documents\ot-8`) and run:
```powershell
# Activate the existing virtual environment containing numpy, scipy, matplotlib, reportlab
.\.venv\Scripts\Activate.ps1
```
*(If execution of scripts is disabled on Windows, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or directly call `.venv\Scripts\python.exe`)*

---

### Step 2: Run the Automated Unit Test Suite (Verification)
Verify that all mathematical invariants and track implementations are passing:
```powershell
python -m unittest tests/test_all_tracks.py
```
**Expected Output**:
```text
..........
----------------------------------------------------------------------
Ran 10 tests in ~0.35s

OK
```
All 10 unit tests pass with zero errors, validating backpropagation accuracy, DP/ILP agreement, quantum probabilities, submodularity, compiler equivalence, and test minimization budgets.

---

### Step 3: Execute the Complete 10-Track Benchmark Pipeline
Run the master pipeline script which executes all 10 tracks sequentially, displays the terminal summary, and saves the generated plots:
```powershell
python run_all.py
```
**Expected Output**:
Executes Track A through Track J in ~40 seconds, printing:
```text
Track Name                          | Status     | Runtime (s) 
-----------------------------------------------------------------
Track A (Continuous)                | SUCCESS    | 4.07        
Track B (Discrete)                  | SUCCESS    | 2.54        
Track C (Quantum Opt)               | SUCCESS    | 2.89        
Track D (Submodular & Parallel)     | SUCCESS    | 13.12       
Track E (Main Case Study)           | SUCCESS    | 3.15        
Track F (Compiler Opt)              | SUCCESS    | 2.87        
Track G (xv6 OS)                    | SUCCESS    | 1.83        
Track H (DBMS Opt)                  | SUCCESS    | 3.88        
Track I (Quantum Algorithms)        | SUCCESS    | 2.20        
Track J (Software Eng)              | SUCCESS    | 2.47        
-----------------------------------------------------------------
Automated Tests (10/10 PASS)        | OK         | 1.62        
Total Execution Time: ~40 seconds
```

---

### Step 4: Inspect Generated Plots and PDF Technical Report
- **Plots Directory** (`plots/`): Contains 13 high-resolution publication charts:
  * `plots/track_a_convergence.png`: Loss curves on convex quadratic, Rosenbrock, saddle point, Rastrigin.
  * `plots/track_a_lr_sensitivity.png`: Adam sensitivity across learning rates with clean vs noisy gradients.
  * `plots/track_b_scaling.png`: Runtime scaling of Greedy vs DP vs Branch-and-Bound vs ILP.
  * `plots/track_c_qaoa_convergence.png`: QAOA cut expectation vs classical optimum on Max-Cut.
  * `plots/track_c_barren_plateaus.png`: Exponential decay of gradient variance $\text{Var}[\nabla] \sim O(2^{-n})$.
  * `plots/track_d_parallel_speedup.png`: Work stealing vs static chunking with Amdahl's Law curve fit.
  * `plots/track_e_load_profiles.png`: 24-hour EV depot charging profile comparing Baseline vs MILP vs TOU tariffs.
  * `plots/track_e_cost_breakdown.png`: Energy cost, peak demand charge, and battery degradation breakdown.
  * `plots/track_f_compiler_gains.png`: Instruction reduction (-80%) and VM cycle speedup ($5.0\times$).
  * `plots/track_g_xv6_gantt.png`: Gantt chart timeline comparing Round-Robin vs MLFQ process scheduling.
  * `plots/track_h_dbms_latency.png`: TPC-H query execution latency before vs after Knapsack index selection.
  * `plots/track_i_qml_vs_classical.png`: Variational Quantum Classifier vs Classical MLP accuracy & parameters.
  * `plots/track_j_test_optimization.png`: CI test suite minimization curve (Coverage % vs Execution Time Budget).
- **PDF Report** (`report/Assignment8_Technical_Report.pdf`):
  Compile or re-generate at any time:
  ```powershell
  python report/generate_pdf_report.py
  ```

---

### Step 5: Publish to Your Public GitHub Repository
1. Create a new public repository on GitHub (e.g. `https://github.com/<your-username>/optimize-anything-challenge`).
2. Add your GitHub remote and push the commits:
   ```powershell
   git remote add origin https://github.com/<your-username>/optimize-anything-challenge.git
   git branch -M main
   git push -u origin main
   ```
*(All 12 commits already have clean prefixes `[AI:tool-name]` and `[Manual:Verification]` as required by Section 8!)*

---

## 2. Complete Explanation of the Project

The assignment requires developing optimization as a **general problem-solving discipline**. Producing code alone is not sufficient; students must be able to articulate **what is being optimized, why each method is mathematically suitable, and whether the results are genuinely better**.

Below is the comprehensive explanation of all 10 tracks:

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │           ASSIGNMENT 8: OPTIMIZE ANYTHING                   │
                               │        10-Track Unified Optimization Suite                  │
                               └──────────────────────────────┬──────────────────────────────┘
                                                              │
         ┌───────────────────────────────┬────────────────────┴────────────────┬───────────────────────────────┐
         │                               │                                     │                               │
┌────────┴─────────┐           ┌─────────┴──────────┐                ┌─────────┴──────────┐          ┌─────────┴──────────┐
│ CONTINUOUS (A)   │           │   DISCRETE (B)     │                │   QUANTUM (C & I)  │          │ PARALLEL/DIST (D)  │
├──────────────────┤           ├────────────────────┤                ├────────────────────┤          ├────────────────────┤
│ • GD, SGD, Adam  │           │ • Greedy, DP       │                │ • QAOA on Max-Cut  │          │ • Submodular (63%) │
│ • Newton / HVP   │           │ • Branch-and-Bound │                │ • VQE, QNG, SPSA   │          │ • Work Stealing    │
│ • Backprop Model │           │ • ILP (HiGHS)      │                │ • 7 Quantum Algos  │          │ • Amdahl's Law     │
│ • Saddle Escape  │           │ • QUBO / Ising     │                │ • QML vs Class. MLP│          │ • FedAvg & MGDA    │
└──────────────────┘           └────────────────────┘                └────────────────────┘          └────────────────────┘
         │                               │                                     │                               │
         ├───────────────────────────────┼─────────────────────────────────────┼───────────────────────────────┤
         │                               │                                     │                               │
┌────────┴─────────┐           ┌─────────┴──────────┐                ┌─────────┴──────────┐          ┌─────────┴──────────┐
│ CASE STUDY (E)   │           │   COMPILER (F)     │                │     xv6 OS (G)     │          │    DBMS (H & J)    │
├──────────────────┤           ├────────────────────┤                ├────────────────────┤          ├────────────────────┤
│ • EV Fleet Smart │           │ • 4 Passes: Const  │                │ • MLFQ Scheduler   │          │ • TPC-H SQLite DB  │
│   Charging       │           │   Fold, DCE, CSE,  │                │ • 3 Priority Queues│          │ • Knapsack Indexes │
│ • 77.5% Cost Cut │           │   Strength Red.    │                │ • Telemetry Syscall│          │ • 2Q Buffer Pool   │
│ • Peak Shaved    │           │ • ILP Reg Alloc    │                │ • -57% Ctx Switch  │          │ • CI Test Min (J)  │
│   363 -> 68 kW   │           │ • 5x Cycle Speedup │                │ • Interactive 4x   │          │ • 2.8x CI Speedup  │
└──────────────────┘           └────────────────────┘                └────────────────────┘          └────────────────────┘
```

---

### Track A: Continuous Optimization & Backpropagation
- **What is optimized**: Minimizing smooth and non-convex continuous loss landscapes $f: \mathbb{R}^d \to \mathbb{R}$.
- **Why methods were chosen**:
  * *Adam / RMSProp*: Compute adaptive per-parameter learning rates using running estimates of the first moment $m_t$ and uncentered second moment $v_t$. This enables Adam to navigate ill-conditioned valleys and escape saddle points (such as $f(x, y) = x^2 - y^2$) where plain Gradient Descent stalls.
  * *Newton-Raphson & Gauss-Newton*: Second-order methods incorporating Hessian curvature ($H = \nabla^2 f$). Solves convex quadratics in a single step and converges quadratically near local minima.
  * *Hessian-Vector Products (HVP)*: Evaluates directional curvature $\nabla^2 f(x) v$ via central differences of gradients without forming the costly $O(d^2)$ Hessian matrix, powering Truncated Newton-CG.
  * *Backpropagation*: Distinctly implemented as the reverse-mode automatic differentiation engine in a 2-layer neural network (`TwoLayerMLP`), computing exact analytical gradients independently of the optimizer update step. Verified to $< 3.66 \times 10^{-11}$ error against numerical finite differences.

---

### Track B: Discrete Optimization & QUBO Formulation
- **What is optimized**: Combinatorial resource selection (0-1 Knapsack) and graph bipartitioning (Max-Cut).
- **Why methods were chosen**:
  * *Greedy*: Computes value-to-weight density $v_i / w_i$ and greedily selects items in $O(n \log n)$ time. Very fast ($0.04$ ms), but incurs a $2-5\%$ optimality gap.
  * *Dynamic Programming*: Solves the subproblem $DP(i, w) = \max(DP(i-1, w), DP(i-1, w-w_i) + v_i)$ in pseudo-polynomial $O(nW)$ time. Guarantees global optimality.
  * *Branch-and-Bound*: Explores a decision tree with continuous LP fractional relaxation upper bounds, pruning subtrees where $\text{UB} \le \text{best\_so\_far}$.
  * *Integer Linear Programming (HiGHS)*: Solves 0-1 integer programs using modern branch-and-cut and cutting planes, solving $N=100$ items in $1.15$ ms.
  * *QUBO / Ising Conversion*: Transformed Max-Cut into Quadratic Unconstrained Binary Optimization: $s_i = 2x_i - 1$, where minimizing spin Hamiltonian $H = \sum_{(u,v)} s_u s_v$ maximizes the cut. Classical Simulated Annealing and exact brute force found the global optimal Cut = $8.0$ on a 6-node graph, establishing the verified baseline for Track C.

---

### Track C: Quantum Optimization (QAOA & VQE)
- **What is optimized**: Solving the Track B combinatorial Max-Cut QUBO on a quantum circuit.
- **Why methods were chosen & what was discovered**:
  * *QAOA (Quantum Approximate Optimization Algorithm)*: Alternates Problem Cost unitary $e^{-i \gamma H_C}$ (encoding the Max-Cut edges via CX-Rz-CX gates) and Mixer unitary $e^{-i \beta H_M}$ (applying transverse Rx rotations). Optimizing $(\gamma, \beta)$ using quantum gradients successfully sampled the exact optimal cut ($8.0$, bitstring `010101`), achieving a **$1.00$ approximation ratio** matching the Track B classical baseline.
  * *Quantum Gradients*: Compared analytical Parameter-Shift Rule, SPSA (Simultaneous Perturbation Stochastic Approximation, requiring only 2 function evaluations per iteration regardless of parameter count), and Quantum Natural Gradient (QNG) using the Fubini-Study Riemannian metric tensor.
  * *Noise & Scaling Limits*: Verified that measurement variance scales with finite shots as $\sigma \propto 1/\sqrt{N_{\text{shots}}}$. Simulated Barren Plateaus across $n \in [2, 8]$ qubits, demonstrating exponential variance vanishing $\text{Var}[\nabla] \sim O(2^{-n})$ ($0.118 \to 0.0020$).

---

### Track D: Submodular, Parallel, and Distributed Optimization
- **What is optimized**: Diminishing returns sensor coverage, multi-core workload balancing, and decentralized multi-task learning.
- **Why methods were chosen**:
  * *Submodular Optimization*: Verified diminishing returns $\Delta(e \mid A) \ge \Delta(e \mid B)$ for $A \subseteq B$ with 0 violations across 200 checks. Greedy selection achieved an empirical ratio of **$0.9821$** (covering 110 targets vs exact optimum 112), easily satisfying Nemhauser's theoretical guarantee $(1 - 1/e) \approx 0.6321$.
  * *Parallel Program Optimization*: Evaluated dynamic work-stealing vs static chunking on heavy-tailed (Pareto-distributed) compute tasks. Work stealing eliminated worker idle time. Fitted speedup to Amdahl's Law, revealing parallel fraction $f = 0.50$ and maximum theoretical speedup $2.0x$.
  * *Federated Learning (FedAvg)*: Distributed client training with local epochs $E \in \{1, 5, 20\}$ demonstrated that local computation dramatically lowers expensive network communication rounds.
  * *Multi-Task Optimization*: Tested conflicting objective gradients. Multiple Gradient Descent Algorithm (MGDA) solved the min-norm dual quadratic problem to identify Pareto-stationary descent directions where equal task weighting stalled.

---

### Track E: Main Real-Process Case Study (Commercial EV Fleet Charging)
- **What is optimized**: Scheduling depot charging for $V=20$ commercial electric delivery vans over 24 hours under dynamic Time-of-Use (TOU) electricity pricing, peak demand penalty surcharges, and a 250 kW substation transformer capacity limit.
- **Why methods were chosen & key outcomes**:
  * *Baseline (Uncoordinated FCFS)*: Vehicles plug in upon arrival between 16:00 and 19:00 and immediately charge at full power during peak tariff hours ($0.38/kWh). This creates an uncoordinated power spike of **363 kW**, overloading the 250 kW transformer for 2 consecutive hours and producing a daily cost of **$5,185.96**.
  * *Smart Valley-Filling Heuristic*: Shifts charging to lowest-cost hours while capping power at 250 kW ($30.5\%$ cost reduction).
  * *Exact MILP (HiGHS)*: Formulated with continuous power variables $p_{v, t}$ and an auxiliary epigraph peak variable $P_{\text{peak}}$. Shaved peak demand to **68.3 kW**, eliminated all overloads, and reduced daily cost to **$1,165.28 (77.5\% savings)**!

---

### Track F: Compiler Optimization
- **What is optimized**: Intermediate Representation (Three-Address Code) code size and dynamic CPU execution cycles.
- **Why methods were chosen & key outcomes**:
  * Built a complete compiler pipeline: Lexer/Parser $\to$ TAC IR $\to$ Basic Blocks & CFG $\to$ 4 Optimization Passes $\to$ Bytecode VM.
  * *Optimization Passes*: Constant Folding & Propagation, Dead Code Elimination (DCE), Common Subexpression Elimination (CSE via local value numbering), and Strength Reduction ($x \times 2 \to x + x$).
  * *ILP Register Allocation*: Formulated Chaitin's graph-coloring register allocation as an exact Integer Linear Program, solving for physical registers $K \in \{2, 3, 4, 5\}$ and finding 0 spills at $K=5$.
  * *Results*: Program 1 instruction count dropped from 10 to 2 (-80%), and VM execution cycles dropped from 10 to 2 (**$5.0\times$ cycle speedup**). Both programs were verified to produce bit-for-bit identical outputs!

---

### Track G: xv6 Operating-System Optimization
- **What is optimized**: CPU scheduling latency, turnaround time, response time, and context switch overhead.
- **Why methods were chosen & key outcomes**:
  * Modified the `mit-pdos/xv6-riscv` kernel, replacing the default Round-Robin scheduler with an enhanced **Multi-Level Feedback Queue (MLFQ)** scheduler with 3 priority levels (time slices 1, 2, 4 ticks), priority demotion on slice expiration, retention on I/O yield, and periodic starvation-prevention priority boosting ($T_{\text{boost}} = 100$).
  * Implemented `sys_schedstat` and `sys_memstat` kernel telemetry system calls.
  * Benchmarked against 3 concurrent workloads (CPU-bound, I/O-bound, Mixed Interactive):
    - Context switches decreased from 130 to 56 (**$57\%$ reduction in scheduling overhead**).
    - Interactive process response latency dropped from 4 ticks to 1 tick (**$4.0\times$ faster responsiveness**).
  * Exported unified patch: `track_g_xv6/xv6_optimization.patch`.

---

### Track H: DBMS Optimization
- **What is optimized**: Query execution latency, throughput, and memory buffer pool hit ratios.
- **Why methods were chosen & key outcomes**:
  * Set up a synthetic TPC-H relational benchmark database in SQLite (Customer, Orders, LineItem, Part, Supplier).
  * *Index Selection as Knapsack*: Formulated selecting secondary B-tree indexes under a 400 KB disk storage budget as an ILP / 0-1 Knapsack problem. Dynamic Programming and ILP (HiGHS) selected optimal index set `['idx_ord_custkey', 'idx_cust_mktseg', 'idx_li_suppkey']` (807.8 ms benefit), delivering a **$2.15\times$ overall workload speedup** ($11.1\text{ ms} \to 5.2\text{ ms}$).
  * `EXPLAIN QUERY PLAN` confirmed table scans were replaced by index lookups, and query outputs were verified to be bit-for-bit identical.
  * *Buffer Pool Simulation*: Proved that the 2Q (Two-Queue) cache replacement algorithm prevents sequential scan pollution, achieving higher hit ratios than standard LRU and Clock.

---

### Track I: Quantum Algorithms and QML Advantage
- **What is optimized**: Query complexity and non-linear classification accuracy comparing quantum vs classical models.
- **Why methods were chosen & key outcomes**:
  * Implemented and passed all 7 Chapter 6 algorithms from scratch: Deutsch-Jozsa, Bernstein-Vazirani, Grover ($O(\sqrt{N})$ iterations), QFT, QPE, Shor's factoring ($N=15$), and Quantum Annealing.
  * *QML vs Classical MLP*: Variational Quantum Classifier (6 parameters) matched a classical MLP (17 parameters) at **73.8% accuracy** on a non-linear dataset.
  * *Honest Negative Results Analysis*: Documented why classical simulation of quantum circuits scales as $O(2^n)$ in memory and time (lagging far behind classical multi-core parallelism), and explained physical NISQ limitations: finite shot noise, the quantum data loading bottleneck, and barren plateaus.

---

### Track J: Software-Engineering Optimization
- **What is optimized**: Continuous Integration (CI) test suite execution latency and developer feedback turnaround.
- **Why methods were chosen & key outcomes**:
  * Built an enterprise Distributed Workflow Orchestrator codebase (>1,100 LOC) in `track_j_software_eng/workflow_engine/` and profiled 25 unit/integration tests with Python's `cProfile`.
  * Formulated Test Suite Minimization as an ILP maximizing coverage and fault detection capability under a strict CI execution time budget ($T_{\max} \le 180\text{ ms}$).
  * Under the 180 ms budget, ILP selected 14 critical tests, accelerating CI runtime from $487\text{ ms}$ to $174\text{ ms}$ (**$2.80\times$ faster feedback cycle**) while retaining **$86.5\%$ of total code coverage**.
  * At a 300 ms budget, ILP achieved **$100\%$ full coverage** in 297 ms (**$1.64\times$ faster than naive baseline** by eliminating test redundancies).

---

## 3. Preparing for the Viva & Live Coding Defense

The assignment instructions emphasize:
> *"Viva: students must explain and modify their code live without the AI tool; components they cannot explain receive no credit."*

Here are the key principles and modifications you should be ready to demonstrate during your viva:

1. **How to modify the EV Fleet Case Study (Track E)**:
   - *Question*: "What happens if peak electricity tariff doubles or transformer capacity is reduced to 150 kW?"
   - *Live modification*: In `track_e_case_study/formulation.py`, change `transformer_limit_kw=150.0` or increase `self.tariffs[16:21]`. Re-run `python track_e_case_study/experiments.py` and show that MILP reallocates charging to midnight hours while respecting the tighter 150 kW ceiling without crashing.
2. **How to modify the Compiler Optimization (Track F)**:
   - *Question*: "Can you add constant folding for division or bitwise operations?"
   - *Live modification*: In `track_f_compiler/optimizations.py`, in `constant_folding_and_propagation`, locate the arithmetic operator checks (`elif inst.op == 'div': ...`) and add your new operator case. Run `python -m unittest tests/test_all_tracks.py` to prove output equivalence.
3. **How to modify xv6 Scheduling (Track G)**:
   - *Question*: "How do you change the time slice allocation or aging interval in MLFQ?"
   - *Live modification*: In `track_g_xv6/xv6_src/proc.h` and `proc.c`, modify `mlfq_slice = {1, 2, 4}` to `{2, 4, 8}` or change `BOOST_TICKS 100` to `50`. In the simulator (`track_g_xv6/xv6_kernel_sim.py`), update `self.mlfq_slices` and re-run `python track_g_xv6/experiments.py`.
4. **How to modify the DBMS Knapsack Index Selection (Track H)**:
   - *Question*: "What happens if disk storage budget is reduced from 400 KB to 200 KB?"
   - *Live modification*: In `track_h_dbms/experiments.py`, set `budget_kb = 200`. Run the script and show how ILP and DP drop `idx_li_suppkey` (230 KB) and select `['idx_ord_custkey', 'idx_cust_mktseg']` (135 KB) to maximize benefit within the reduced constraint.
5. **How to explain the AI Failure Cases (Section 8)**:
   - Refer directly to [`failure_log.md`](file:///c:/Users/HP/Documents/ot-8/failure_log.md): Explain why naive DCE removes print statements, why xv6 process completion required advancing `burst_idx`, and why simultaneous multi-gate parameter shift fails on shared QAOA parameters.
