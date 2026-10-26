# Prompt Log - Track B: Discrete Optimization

## 1. Original User Prompt
"Implement Track B discrete optimization methods: Greedy, Dynamic Programming, Branch-and-Bound, and Integer Linear Programming (ILP). Also formulate a combinatorial problem (Max-Cut) as a QUBO/Ising model with binary-to-spin conversion and penalty terms, and solve it classically with Simulated Annealing and exact enumeration as the baseline for Track C."

## 2. AI Assistant Response Excerpt
Created `track_b_discrete/solvers.py` with 0-1 Knapsack solvers (Greedy, DP, custom Branch-and-Bound with fractional LP relaxation pruning, and `scipy.optimize.milp` HiGHS solver).
Created `track_b_discrete/qubo_ising.py` with `QUBOInstance`, binary-to-spin transformations $s_i = 2x_i - 1$, Max-Cut QUBO formulation, and classical solvers (Metropolis Simulated Annealing and Exact Brute Force).

## 3. Critical Review & Revised Prompt
"Check for escape character warnings in docstrings and ensure the exact optimal cut from classical QUBO is serialized to JSON so Track C can directly ingest it as a baseline."

## 4. Resulting Code Change
- Fixed regex docstring raw formatting in `qubo_ising.py`.
- Serialized graph edges, QUBO matrix, and exact classical cut ($8.0$) to `results/track_b_qubo_baseline.json`.

## 5. Verification Note
- Exact classical brute-force and Simulated Annealing both converged to optimal Cut = 8.0 on the 6-vertex benchmark graph.
- Runtime scaling across $N \in [10, 20, 30, 50, 100]$ saved in `plots/track_b_scaling.png`.
