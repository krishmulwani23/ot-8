# Prompt Log - Track E: Main Case Study (Electric Vehicle Fleet Charging)

## 1. Original User Prompt
"Implement Track E Main Real-Process Case Study: choose an open domain (Smart Commercial EV Fleet Dispatch & Charging under Dynamic TOU Tariffs and Peak Demand Charges), follow the Section 4 workflow, build an Uncoordinated Baseline, an Exact MILP solver (HiGHS), and a Smart Valley-Filling Heuristic, and evaluate cost, peak demand, and battery degradation."

## 2. AI Assistant Response Excerpt
Created `track_e_case_study/formulation.py`, `track_e_case_study/baseline.py`, `track_e_case_study/solver_milp.py`, and `track_e_case_study/solver_heuristic.py`.
Modeled $V=20$ commercial delivery vans over 24 hours ($480$ power variables + $1$ auxiliary peak power variable $P_{\text{peak}}$). Formulated epigraph constraints to linearize peak demand charges.

## 3. Critical Review & Revised Prompt
"Ensure the transformer capacity limit constraint is strictly enforced across all 24 hours and verify that the uncoordinated baseline violates transformer capacity while MILP eliminates overloads."

## 4. Resulting Code Change
- Formulated linear constraints: $\sum_v p_{v, t} \le P_{\text{peak}}$ and $\sum_v p_{v, t} \le P_{\text{transformer}}$ for all $t$.
- Plotted dual-axis load profile and TOU tariff schedule in `plots/track_e_load_profiles.png`.

## 5. Verification Note
- Baseline Cost: $5,185.96 (overloaded transformer by up to 363 kW for 2 hours).
- Smart Heuristic Cost: $3,604.18 (30.5% savings, zero overloads).
- MILP Optimal Cost: $1,165.28 (77.5% massive savings, smoothed peak demand to 68.3 kW).
