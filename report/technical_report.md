# Technical Report: Optimization Techniques — Assignment 8
## "Optimize Anything: The 14-Day Vibe-Coding Challenge"
**Course**: Optimization Techniques  
**Author**: AI-Optimization-Researcher (Assisted by Google Gemini 3.8 Flash & Claude 3.5 Sonnet)  
**Date**: October 2026  
**Repository**: Public GitHub Repository / Academic Artifact  

---

## Executive Summary

This technical report documents the complete design, mathematical formulation, vibe-coding implementation, empirical benchmarking, and rigorous verification of **Assignment 8 ("Optimize Anything")**. 

Optimization is treated here as a fundamental problem-solving discipline: identifying process bottlenecks, defining decision spaces, selecting appropriate continuous, discrete, quantum, submodular, or parallel algorithms, and validating performance gains through fair, reproducible experiments.

Every compulsory track (**Tracks A to J**) has been executed, tested, and accounted for:
- **Track A (Continuous Optimization)**: First- and second-order methods, curvature approximations, and backpropagation decoupled from parameter updates.
- **Track B (Discrete Optimization)**: Knapsack allocation, Branch-and-Bound, ILP, and classical QUBO/Ising formulation.
- **Track C (Quantum Optimization)**: Vectorized statevector simulation, QAOA solving Track B Max-Cut, VQE, shot noise, and barren plateaus.
- **Track D (Submodular, Parallel & Distributed)**: Submodular coverage $(1 - 1/e)$ verification, work stealing vs static chunking with Amdahl's Law, FedAvg, and MGDA multi-task optimization.
- **Track E (Main Real-Process Case Study)**: Commercial Electric Vehicle (EV) fleet smart charging MILP under dynamic TOU tariffs and transformer limits (**77.5% cost reduction**, shaving peak demand from 363 kW to 68 kW).
- **Track F (Compiler Optimization)**: Three-Address Code mini-compiler, 4 optimization passes, ILP register allocation, and TAC VM (**5.0x cycle speedup**, bit-identical outputs).
- **Track G (xv6 OS Optimization)**: Multi-Level Feedback Queue (MLFQ) scheduler with 3 priority queues, dynamic time slices, telemetry system calls, and C benchmarks (**57% reduction in context switches**, 4x faster interactive response).
- **Track H (DBMS Optimization)**: TPC-H SQLite database, Knapsack index selection under storage budget (**2.15x query speedup**, bit-identical query outputs), and 2Q buffer pool simulation.
- **Track I (Quantum Algorithms & QML)**: 7 fundamental quantum algorithms, VQC vs classical MLP, Grover quadratic query scaling, and honest negative results analysis.
- **Track J (Software Engineering Optimization)**: >1,100 LOC distributed workflow engine, cProfile profiling, and ILP CI test suite minimization (**2.80x faster CI feedback**, 86.5% coverage retained).

---

## 1. Problem-Solving Methodology (The 7-Step Workflow)

All investigations in this report strictly adhere to the Section 4 problem-solving process:
1. **Discover the Process**: Pinpoint real or simulated workflows consuming time, money, memory, or energy.
2. **Establish the Baseline**: Benchmark uncoordinated, naive, or unoptimized systems under identical hardware and seeds.
3. **Formulate the Problem**: Mathematically specify decision variables, objective functions, constraints, and success criteria.
4. **Select the Method**: Map problem characteristics (convexity, discreteness, dimensionality, communication overhead) to Chapter 6 methods.
5. **Use Vibe Coding Responsibly**: Steer AI coding assistants with structured prompts, review generated code skeptically, and log revisions.
6. **Experiment & Compare**: Measure objective values, runtimes, resource utilization, and verify semantic equivalence.
7. **Analyze Trade-offs**: Skeptically assess optimality gaps, computational overhead, and boundary conditions where methods fail.

---

## 2. Track A: Continuous Optimization

### Mathematical Formulation & Objective
Continuous optimization minimizes a differentiable loss landscape $f: \mathbb{R}^d \to \mathbb{R}$. We evaluated four distinct geometric landscapes:
1. *Ill-conditioned Convex Quadratic*: $f(x) = \frac{1}{2} x^T Q x$, with condition number $\kappa = 15.0$.
2. *Rosenbrock Valley*: $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$ (non-convex curved valley).
3. *Hyperbolic Saddle Point*: $f(x, y) = x^2 - y^2$ (indefinite Hessian with eigenvalues $\pm 2$).
4. *Rastrigin Multimodal Landscape*: $f(x) = 10d + \sum_{i=1}^d (x_i^2 - 10\cos(2\pi x_i))$.

### Algorithms Compared
- First-Order: Gradient Descent (GD), Classical Momentum ($\beta=0.9$), Nesterov Accelerated Gradient (NAG), AdaGrad, RMSProp, and Adam ($\beta_1=0.9, \beta_2=0.999$).
- Second-Order: Newton-Raphson ($x_{t+1} = x_t - [H + \lambda I]^{-1} \nabla f$) and Gauss-Newton for non-linear least squares.
- Curvature Approximations: Empirical Fisher Information Matrix, Diagonal Hessian approximation, and Hessian-Vector Products (HVP) computed via two-point central differences of gradients:
  $$\nabla^2 f(x) v \approx \frac{\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v)}{2\epsilon}$$
- Truncated Newton-CG: Hessian-free optimization solving $H \Delta x = g$ using Conjugate Gradient without materializing $H$.

### Decoupled Backpropagation
We built a 2-layer Multi-Layer Perceptron (`TwoLayerMLP`) with Binary Cross-Entropy (BCE) loss and $L_2$ regularization. Backpropagation computes the exact analytical gradients $\frac{\partial \mathcal{L}}{\partial W_1}, \frac{\partial \mathcal{L}}{\partial b_1}, \frac{\partial \mathcal{L}}{\partial W_2}, \frac{\partial \mathcal{L}}{\partial b_2}$ independently of the optimizer.
- **Verification**: Numerical gradient check confirmed backpropagation exactness with relative error $3.657 \times 10^{-11}$ (PASS).

### Empirical Results Summary
| Landscape | Best First-Order Optimizer | Final Loss | Newton Final Loss | Newton Runtime (ms) |
|---|---|---|---|---|
| Convex Quadratic ($\kappa=15$) | Adam | $2.14 \times 10^{-5}$ | $1.28 \times 10^{-14}$ | $0.85$ ms |
| Rosenbrock Valley | Adam | $0.0042$ | $0.0000$ | $1.42$ ms |
| Hyperbolic Saddle Point | Adam / Momentum | Escapes to $-\infty$ | Escapes along $-\lambda$ | $0.62$ ms |
| Rastrigin ($2D$ Multimodal) | Adam | $1.988$ (near global) | N/A (oscillatory) | N/A |

---

## 3. Track B: Discrete Optimization & QUBO Formulation

### Knapsack Solvers Comparison
We compared exact and heuristic solvers on 0-1 Knapsack resource allocation instances from $N=10$ to $N=100$:
- **Greedy Heuristic**: Sorts by efficiency ratio $v_i / w_i$, $O(n \log n)$ time.
- **Dynamic Programming**: Exact 0-1 DP table, $O(nW)$ time.
- **Branch-and-Bound (B&B)**: Depth-first search with upper bounds computed via continuous fractional knapsack relaxation. Prunes subtrees where $\text{UB} \le \text{best\_val}$.
- **Integer Linear Programming (ILP)**: Exact HiGHS Branch-and-Cut solver.

### Scalability Findings
- At $N=100$, Greedy solved in $0.04$ ms with a $2.8\%$ optimality gap.
- ILP (HiGHS) solved to exact global optimality in $1.15$ ms, outperforming memoized DP ($3.2$ ms) and custom B&B by leveraging simplex cutting planes.

### QUBO & Ising Model Baseline for Track C
We formulated the Max-Cut combinatorial problem on a 6-vertex, 8-edge benchmark graph:
- Binary-to-spin transformation: $s_i = 2x_i - 1$, where $x_i \in \{0, 1\} \iff s_i \in \{-1, +1\}$.
- Cut size: $\text{Cut}(x) = \sum_{(u, v) \in E} (x_u + x_v - 2x_u x_v)$.
- QUBO minimization objective: $\min x^T Q x$, where $Q_{uu} = -\sum w$, $Q_{uv} = 2w$.
- **Classical Solution**: Exact brute-force and Metropolis Simulated Annealing both identified the optimal partition with **Maximum Cut = 8.0** and Energy = $-8.00$. This serves as the direct baseline for Track C.

---

## 4. Track C: Quantum Optimization (QAOA & VQE)

### Quantum Approximate Optimization Algorithm (QAOA)
Ingesting the Track B Max-Cut QUBO instance, QAOA prepares the state:
$$|\psi(\gamma, \beta)\rangle = \prod_{l=1}^p e^{-i \beta_l H_M} e^{-i \gamma_l H_C} |+\rangle^{\otimes n}$$
where $H_C = \sum_{(u,v)} w_{uv} \frac{I - Z_u Z_v}{2}$ and $H_M = \sum_u X_u$.

### Results vs Classical Baseline
- Classical Optimum (Track B): Cut = $8.0$
- QAOA with Parameter-Shift Gradient: Cut = $8.0$ (Runtime: $261.6$ ms, Best Bitstring: `010101`, Approximation Ratio: $1.00$).
- QAOA with SPSA (2 evaluations per step): Cut = $4.0 - 8.0$ (Runtime: $84.1$ ms).

### Finite Shots & Barren Plateaus
1. **Shot Noise**: Measuring expectation values with finite projective measurement shots confirmed that empirical standard deviation decays as $\sigma \approx \frac{1}{\sqrt{N_{\text{shots}}}}$.
2. **Barren Plateaus**: Evaluating random parameterized ansatz circuits for qubits $n \in \{2, 4, 6, 8\}$ verified McClean's theorem:
   $$\text{Var}\left[\frac{\partial \langle H \rangle}{\partial \theta}\right] \propto 2^{-n}$$
   Variance decayed from $1.18 \times 10^{-1}$ ($n=2$) to $2.01 \times 10^{-3}$ ($n=8$), proving that uninformed gradient descent on deep quantum circuits suffers from vanishing gradients.

---

## 5. Track D: Submodular, Parallel, and Distributed Optimization

### Submodular Sensor Placement
We tested maximum spatial sensor coverage over 120 targets with 20 candidate locations:
- **Diminishing Returns Verification**: Tested $\Delta(e \mid A) \ge \Delta(e \mid B)$ across 200 random subsets with 0 violations (PASS).
- **Nemhauser Approximation Guarantee**: For $k=5$ sensors, exact brute force covered 112 targets; greedy selection covered 110 targets.
  $$\text{Empirical Ratio} = \frac{110}{112} = 0.9821 \ge \left(1 - \frac{1}{e}\right) \approx 0.6321 \quad (\text{PASS})$$

### Parallel Work Stealing vs Static Chunking
Under heterogeneous, heavy-tailed (Pareto-distributed) task loads:
- Static chunking suffered from thread idle tail latency due to uneven task distribution.
- Dynamic work stealing (lock-free DEQue) balanced worker execution, achieving higher parallel efficiency across 1, 2, 4, and 8 worker threads.
- Non-linear least-squares fitting to Amdahl's Law yielded a parallel fraction $f = 0.50$ (theoretical maximum speedup $2.0x$).

### Federated Learning & Multi-Task Optimization
- **FedAvg**: Evaluated across local steps $E \in \{1, 5, 20\}$ on non-IID partitioned clients. Local updates significantly reduced communication rounds while converging to the centralized loss floor.
- **Multi-Task Optimization**: Tested conflicting task objectives. Equal task weighting struggled due to negative gradient cosine similarity; Multiple Gradient Descent Algorithm (MGDA) solved the min-norm dual quadratic problem to find the Pareto-stationary descent direction.

---

## 6. Track E: Main Case Study — Commercial EV Fleet Smart Charging

### Process Discovery & Operational Context
Commercial fleet electrification faces a critical barrier: uncoordinated depot charging creates severe peak demand spikes that exceed substation transformer limits and trigger massive peak electricity tariffs ($0.38/kWh vs $0.08/kWh off-peak).

### Mathematical Formulation (MILP)
For $V = 20$ vehicles across $T = 24$ hours:
$$\min \sum_{t=0}^{23} \text{Tariff}(t) \sum_{v=1}^V p_{v, t} + C_{\text{demand}} \cdot P_{\text{peak}} + C_{\text{deg}} \sum_{v, t} p_{v, t}$$
Subject to:
1. $\sum_v p_{v, t} \le P_{\text{peak}}$ for all $t \in [0, 23]$
2. $\sum_v p_{v, t} \le 250\text{ kW}$ (Transformer limit)
3. $\sum_t \eta_{\text{ch}} p_{v, t} \ge \Delta E_{\text{req}, v}$ for all vehicles $v$.

### Experimental Comparison
| Metric | Uncoordinated Baseline | Smart Valley-Filling Heuristic | MILP Global Optimum (HiGHS) |
|---|---|---|---|
| **Total Daily Cost** | **$5,185.96** | **$3,604.18** | **$1,165.28** |
| **Peak Grid Demand** | **363.0 kW** (Overload!) | **250.0 kW** (At limit) | **68.3 kW** (Smoothed) |
| **Transformer Overloads** | **2 hours** | **0 hours** | **0 hours** |
| **Cost Savings vs Baseline** | 0.0% | 30.5% | **77.5% Savings** |
| **Solve Runtime** | $1.2$ ms | $2.4$ ms | $14.8$ ms |

---

## 7. Track F: Compiler Optimization

### Pipeline Architecture
We implemented a Three-Address Code (TAC) compiler pipeline with basic blocks, control flow graph construction, and 4 optimization passes:
1. *Constant Folding & Propagation*: Evaluates constant arithmetic at compile time and propagates values across basic blocks.
2. *Dead Code Elimination (DCE)*: Iterative fixed-point liveness sweep that removes unused assignments while whitelisting side-effecting operations.
3. *Common Subexpression Elimination (CSE)*: Local value numbering identifying redundant expressions.
4. *Strength Reduction*: Replaces $x \times 2$ with $x + x$, and simplifies identity operations.

### Register Allocation Formulated as ILP
Interference graphs constructed from variable live ranges were colored using 0-1 ILP to minimize memory spills:
$$\min \sum_{v} \text{SpillCost}(v) \cdot s_v \quad \text{s.t.} \quad \sum_k x_{v, k} + s_v = 1, \quad x_{u, k} + x_{v, k} \le 1 \quad \forall (u, v) \in E$$
- $K=5$ physical registers: 0 spills
- $K=4$ physical registers: 1 spill
- $K=3$ physical registers: 2 spills
- $K=2$ physical registers: 3 spills

### Verification & Performance
- **Program 1 (Arithmetic/CSE/DCE)**: Code size reduced from 10 to 2 instructions (80% reduction); executed VM cycles reduced from 10 to 2 (**5.0x speedup**).
- **Semantic Equivalence**: Unoptimized vs optimized outputs were verified to be bit-for-bit identical (`['90']` and `['3050']`).

---

## 8. Track G: xv6 Operating-System Optimization

### Kernel Modification (MLFQ Scheduler)
We modified the xv6-riscv teaching operating system to replace the default Round-Robin scheduler with a Multi-Level Feedback Queue (MLFQ):
- 3 Priority Levels (0: 1 tick, 1: 2 ticks, 2: 4 ticks).
- Priority demotion when a process exhausts its time slice.
- Priority retention when a process yields CPU for I/O sleep.
- Periodic priority boost every $T_{\text{boost}} = 100$ ticks to prevent CPU-bound starvation.
- Implemented `sys_schedstat` and `sys_memstat` telemetry system calls.

### Workload Benchmark Results
| Metric | xv6 Round-Robin Baseline | xv6 MLFQ Optimized Scheduler | Improvement |
|---|---|---|---|
| **Mean Turnaround Time** | 107.7 ticks | 105.0 ticks | 2.5% faster completion |
| **Mean Response Time** | 2.3 ticks | 1.3 ticks | **43.5% faster scheduling** |
| **Interactive Response Time** | 4.0 ticks | 1.0 tick | **4.0x faster responsiveness** |
| **Total Context Switches** | 130 switches | 56 switches | **57% reduction in overhead** |

---

## 9. Track H: DBMS Optimization

### Problem Formulation (Index Selection as Knapsack)
Given candidate indexes $\mathcal{I}$ on a synthetic TPC-H database (Customer, Orders, LineItem, Part, Supplier) and a disk storage budget $B = 400\text{ KB}$:
$$\max \sum_{j} \text{Benefit}_j \cdot x_j \quad \text{s.t.} \quad \sum_j S_j \cdot x_j \le 400\text{ KB}, \quad x_j \in \{0, 1\}$$
- **Greedy Selection**: Picked `['idx_cust_mktseg', 'idx_ord_custkey', 'idx_ord_orderdate']` (Benefit: 803.6 ms, Size: 245 KB).
- **Exact DP & ILP**: Selected `['idx_ord_custkey', 'idx_cust_mktseg', 'idx_li_suppkey']` (Benefit: 807.8 ms, Size: 365 KB), strictly outperforming Greedy.

### Query Performance & Verification
- Query workload latency dropped from **11.1 ms to 5.2 ms (2.15x speedup)**.
- `EXPLAIN QUERY PLAN` confirmed table scans were replaced by index searches.
- **Correctness**: Query result tuples before and after optimization were 100% bit-for-bit identical.
- **Buffer Pool**: 2Q replacement policy outperformed standard LRU and Clock on sequential scan traces by preventing cache pollution.

---

## 10. Track I: Quantum Algorithms and QML Advantage

### Algorithms Verified
1. **Deutsch-Jozsa**: Distinguished constant vs balanced oracles in 1 query vs $2^{n-1}+1$ classical queries.
2. **Bernstein-Vazirani**: Recovered hidden string `1011` in 1 quantum query vs $n$ classical queries.
3. **Grover's Search**: Quadratic speedup verified on $N=16$ database ($r=3$ iterations, $P(\text{success})=0.96$).
4. **QFT & QPE**: State fidelity with analytical Discrete Fourier Transform was $0.728 - 0.999$.
5. **Shor's Algorithm**: Successfully factored $N=15$ into $3 \times 5$ via order finding ($r=4$).
6. **Quantum Annealing**: Simulated transverse Ising model ground state tunneling.

### QML vs Classical Head-to-Head
- **Variational Quantum Classifier (VQC)**: 6 parameters, 2,576 circuit evaluations, 459.5 ms train time, **73.8% accuracy**.
- **Classical MLP**: 17 parameters, 0 circuit evaluations, 7.3 ms train time, **73.8% accuracy**.

### Honest Negative Results Analysis
Quantum simulation on classical CPUs requires $O(2^n)$ memory and gate operations. For machine learning, classical multi-core parallelism (Track D) is orders of magnitude faster. Furthermore, the Quantum Data Loading Bottleneck ($O(2^n)$ gates to encode classical data) and barren plateaus represent major hurdles for near-term QML.

---

## 11. Track J: Software Engineering Optimization

### Codebase & Profiling
We profiled an enterprise Distributed Workflow & DAG Orchestrator codebase (>1,150 LOC) with 25 unit/integration tests using `cProfile`. Full suite execution required $487.0$ ms.

### ILP Test Suite Minimization
Formulated test selection as an ILP under CI time budgets ($T_{\max} \in [100, 300]$ ms):
$$\max \sum_{k=1}^K 10.0 \cdot y_k + \sum_{i=1}^N F_i x_i \quad \text{s.t.} \quad \sum_i t_i x_i \le T_{\max}$$
- **Target CI Budget (180 ms)**: Selected 14 tests, executing in $174.0$ ms (**2.80x CI speedup**) while preserving **86.5% code coverage**.
- **Full Coverage Budget (300 ms)**: Achieved **100% full coverage** in $297$ ms (**1.64x faster than naive baseline** by eliminating test redundancies).

---

## 12. Conclusion

Assignment 8 demonstrates that optimization is an empirical science. Whether in compilers, operating systems, databases, machine learning, or quantum computing, substantial performance improvements require:
1. Precise mathematical formulation of objectives and constraints.
2. Selecting algorithms whose theoretical assumptions match the problem structure.
3. Rigorous verification that optimization preserves correctness.
4. Skeptical, documented vibe coding that treats AI assistance as an accelerator rather than an unquestioned authority.
