"""
Metaheuristic / Smart Valley-Filling Heuristic for Track E.
Prioritizes lowest tariff hours first, enforcing smooth peak shaving.
"""

import time
import numpy as np
from typing import Dict, Any
from track_e_case_study.formulation import FleetChargingProblem

class SmartHeuristicOptimizer:
    def solve(self, problem: FleetChargingProblem) -> Dict[str, Any]:
        t0 = time.perf_counter()
        V = problem.n_vehicles
        T = 24
        power = np.zeros((V, T), dtype=np.float64)

        # For each vehicle, sort available hours by ascending electricity tariff
        # and allocate charging to the lowest price hours while respecting transformer limits
        for v in range(V):
            veh = problem.vehicles[v]
            needed_energy = max(0.0, (veh.target_soc - veh.initial_soc) * veh.capacity_kwh)
            needed_grid_kwh = needed_energy / problem.charge_efficiency

            # Get hours when vehicle is plugged in
            avail_hours = [t for t in range(T) if problem.is_vehicle_plugged_in(v, t)]
            # Sort by tariff ascending
            avail_hours.sort(key=lambda t: problem.tariffs[t])

            rem_kwh = needed_grid_kwh
            for t in avail_hours:
                if rem_kwh <= 0.0:
                    break
                # Check current grid load at hour t
                current_total_load = np.sum(power[:, t])
                grid_headroom = max(0.0, problem.transformer_limit - current_total_load)
                alloc = min(rem_kwh, veh.max_kw, grid_headroom)
                power[v, t] = alloc
                rem_kwh -= alloc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        grid_load = np.sum(power, axis=0)
        energy_cost = float(np.sum(grid_load * problem.tariffs))
        peak_demand_kw = float(np.max(grid_load))
        demand_penalty = peak_demand_kw * problem.demand_charge
        deg_cost = float(np.sum(power)) * problem.deg_cost_per_kwh
        total_cost = energy_cost + demand_penalty + deg_cost

        return {
            "strategy": "Smart_Valley_Filling_Heuristic",
            "total_cost": total_cost,
            "energy_cost": energy_cost,
            "demand_penalty_cost": demand_penalty,
            "degradation_cost": deg_cost,
            "peak_demand_kw": peak_demand_kw,
            "total_energy_kwh": float(np.sum(power)),
            "transformer_overload_hours": int(np.sum(grid_load > problem.transformer_limit)),
            "hourly_load_kw": grid_load.tolist(),
            "runtime_ms": elapsed_ms
        }
