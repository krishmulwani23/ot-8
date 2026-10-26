"""
Uncoordinated Baseline Strategy (Track E):
First-Come-First-Serve (FCFS) immediate plug-in charging.
Vehicles charge at maximum power immediately upon arrival until target SOC is achieved.
"""

import time
import numpy as np
from typing import Dict, Any
from track_e_case_study.formulation import FleetChargingProblem

class UncoordinatedBaseline:
    def solve(self, problem: FleetChargingProblem) -> Dict[str, Any]:
        t0 = time.perf_counter()
        V = problem.n_vehicles
        T = 24
        # power[v, t] in kW
        power = np.zeros((V, T), dtype=np.float64)
        soc = np.zeros((V, T + 1), dtype=np.float64)

        for v_idx, veh in enumerate(problem.vehicles):
            soc[v_idx, 0] = veh.initial_soc
            curr_soc = veh.initial_soc

            # Simulate 24 hours starting from arrival
            # Order of hours for an overnight vehicle: arrival, arrival+1, ... 23, 0, 1, ... departure-1
            for t in range(T):
                is_present = problem.is_vehicle_plugged_in(v_idx, t)
                needed_energy = max(0.0, (veh.target_soc - curr_soc) * veh.capacity_kwh)

                if is_present and needed_energy > 0.0:
                    max_possible = veh.max_kw * problem.charge_efficiency
                    delivered_energy = min(needed_energy, max_possible)
                    # Charger power drawn from grid
                    p_grid = delivered_energy / problem.charge_efficiency
                    power[v_idx, t] = p_grid
                    curr_soc += delivered_energy / veh.capacity_kwh

                soc[v_idx, t + 1] = curr_soc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Compute cost metrics
        grid_load_by_hour = np.sum(power, axis=0) # shape (24,)
        energy_cost = float(np.sum(grid_load_by_hour * problem.tariffs))
        peak_demand_kw = float(np.max(grid_load_by_hour))
        demand_penalty_cost = peak_demand_kw * problem.demand_charge
        total_energy_kwh = float(np.sum(power))
        degradation_cost = total_energy_kwh * problem.deg_cost_per_kwh
        total_cost = energy_cost + demand_penalty_cost + degradation_cost

        # Check transformer overload violations
        overload_hours = int(np.sum(grid_load_by_hour > problem.transformer_limit))

        return {
            "strategy": "Uncoordinated_FCFS_Baseline",
            "total_cost": total_cost,
            "energy_cost": energy_cost,
            "demand_penalty_cost": demand_penalty_cost,
            "degradation_cost": degradation_cost,
            "peak_demand_kw": peak_demand_kw,
            "total_energy_kwh": total_energy_kwh,
            "transformer_overload_hours": overload_hours,
            "hourly_load_kw": grid_load_by_hour.tolist(),
            "runtime_ms": elapsed_ms
        }
