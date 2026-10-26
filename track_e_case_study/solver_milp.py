"""
Exact Mixed-Integer / Linear Programming Solver for Track E (MILP via HiGHS).
Finds the mathematically provable globally optimal charging schedule.
"""

import time
import numpy as np
from typing import Dict, Any
from scipy.optimize import milp, LinearConstraint, Bounds
from track_e_case_study.formulation import FleetChargingProblem

class MILPFleetOptimizer:
    def solve(self, problem: FleetChargingProblem) -> Dict[str, Any]:
        t0 = time.perf_counter()
        V = problem.n_vehicles
        T = 24
        # Variables:
        # p[v, t] for v in range(V), t in range(T) -> V * T variables
        # Followed by P_peak -> 1 variable
        # Total variables: n_vars = V * T + 1
        n_vars = V * T + 1
        p_peak_idx = V * T

        # Objective vector c:
        # min sum_t [ (tariff[t] + deg_cost) * sum_v p[v, t] ] + demand_charge * P_peak
        c = np.zeros(n_vars, dtype=np.float64)
        for v in range(V):
            for t in range(T):
                idx = v * T + t
                c[idx] = problem.tariffs[t] + problem.deg_cost_per_kwh
        c[p_peak_idx] = problem.demand_charge

        # Variable Bounds:
        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.zeros(n_vars, dtype=np.float64)
        for v in range(V):
            veh = problem.vehicles[v]
            for t in range(T):
                idx = v * T + t
                if problem.is_vehicle_plugged_in(v, t):
                    ub[idx] = veh.max_kw
                else:
                    ub[idx] = 0.0 # not at depot
        ub[p_peak_idx] = problem.transformer_limit # P_peak cannot exceed transformer capacity

        bounds = Bounds(lb=lb, ub=ub)

        # Constraints:
        # 1. Peak demand tracking: sum_v p[v, t] - P_peak <= 0  (T constraints)
        # 2. Transformer capacity: sum_v p[v, t] <= P_transformer (T constraints)
        # 3. Vehicle energy fulfillment: sum_t eta * p[v, t] >= (target_soc - init_soc) * cap (V constraints)
        n_constraints = 2 * T + V
        A = np.zeros((n_constraints, n_vars), dtype=np.float64)
        lhs = np.full(n_constraints, -np.inf, dtype=np.float64)
        rhs = np.zeros(n_constraints, dtype=np.float64)

        row = 0
        # 1. Peak tracking: sum_v p[v, t] - P_peak <= 0
        for t in range(T):
            for v in range(V):
                A[row, v * T + t] = 1.0
            A[row, p_peak_idx] = -1.0
            rhs[row] = 0.0
            row += 1

        # 2. Transformer capacity: sum_v p[v, t] <= transformer_limit
        for t in range(T):
            for v in range(V):
                A[row, v * T + t] = 1.0
            rhs[row] = problem.transformer_limit
            row += 1

        # 3. Energy fulfillment: sum_t eta * p[v, t] >= needed_energy
        for v in range(V):
            veh = problem.vehicles[v]
            needed_energy = max(0.0, (veh.target_soc - veh.initial_soc) * veh.capacity_kwh)
            for t in range(T):
                if problem.is_vehicle_plugged_in(v, t):
                    A[row, v * T + t] = problem.charge_efficiency
            lhs[row] = needed_energy
            rhs[row] = np.inf
            row += 1

        constraints = LinearConstraint(A=A, lb=lhs, ub=rhs)
        integrality = np.zeros(n_vars, dtype=int) # continuous LP relaxation solves exactly

        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not res.success:
            raise RuntimeError(f"MILP solve failed: {res.status}")

        x_sol = res.x
        power = x_sol[:V * T].reshape(V, T)
        p_peak = float(x_sol[p_peak_idx])

        grid_load = np.sum(power, axis=0)
        energy_cost = float(np.sum(grid_load * problem.tariffs))
        demand_penalty = p_peak * problem.demand_charge
        deg_cost = float(np.sum(power)) * problem.deg_cost_per_kwh
        total_cost = energy_cost + demand_penalty + deg_cost

        return {
            "strategy": "MILP_Optimal_HiGHS",
            "total_cost": total_cost,
            "energy_cost": energy_cost,
            "demand_penalty_cost": demand_penalty,
            "degradation_cost": deg_cost,
            "peak_demand_kw": float(np.max(grid_load)),
            "total_energy_kwh": float(np.sum(power)),
            "transformer_overload_hours": int(np.sum(grid_load > problem.transformer_limit)),
            "hourly_load_kw": grid_load.tolist(),
            "runtime_ms": elapsed_ms
        }
