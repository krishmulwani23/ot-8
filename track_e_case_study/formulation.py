"""
Problem Formulation for Main Case Study (Track E):
Smart Commercial Electric Vehicle (EV) Fleet Charging Schedule & Peak Demand Optimization
under Dynamic Time-of-Use (TOU) Electricity Tariffs and Battery Health Constraints.
"""

import numpy as np
from typing import List, Dict, Any

class EVVehicle:
    """Represents a single commercial fleet vehicle."""
    def __init__(self, vid: int, battery_capacity_kwh: float, max_charge_kw: float,
                 arrival_hour: int, departure_hour: int,
                 initial_soc: float, target_soc: float = 0.90):
        self.vid = vid
        self.capacity_kwh = battery_capacity_kwh
        self.max_kw = max_charge_kw
        self.arrival = arrival_hour # e.g. 17 (5 PM)
        self.departure = departure_hour # e.g. 7 (7 AM next day)
        self.initial_soc = initial_soc
        self.target_soc = target_soc


class FleetChargingProblem:
    """
    Fleet Charging Problem over a 24-hour planning horizon (t = 0 to 23).
    Includes:
    - Time-of-Use (TOU) electricity pricing:
      * Off-peak (23:00 - 06:00): $0.08 / kWh
      * Mid-peak (06:00 - 16:00): $0.15 / kWh
      * On-peak (16:00 - 21:00): $0.35 / kWh
      * Mid-peak (21:00 - 23:00): $0.18 / kWh
    - Monthly peak demand penalty: $12.00 / kW
    - Battery degradation cost: $0.015 / kWh cycled
    - Depot transformer capacity limit: 250 kW
    """
    def __init__(self, n_vehicles: int = 15, transformer_limit_kw: float = 220.0,
                 demand_charge_per_kw: float = 12.0, seed: int = 42):
        self.n_vehicles = n_vehicles
        self.transformer_limit = transformer_limit_kw
        self.demand_charge = demand_charge_per_kw
        self.deg_cost_per_kwh = 0.015
        self.charge_efficiency = 0.92

        # 24-hour TOU electricity tariff profile ($/kWh)
        self.tariffs = np.array([
            0.08, 0.08, 0.08, 0.08, 0.08, 0.08, # 00:00 - 06:00 (Off-peak)
            0.14, 0.14, 0.16, 0.16, 0.16, 0.16, # 06:00 - 12:00 (Mid-peak morning)
            0.15, 0.15, 0.18, 0.22, 0.35, 0.38, # 12:00 - 18:00 (Ramp & Peak evening)
            0.36, 0.32, 0.25, 0.18, 0.12, 0.09  # 18:00 - 24:00 (Peak decay to off-peak)
        ], dtype=np.float64)

        # Generate realistic commercial fleet (e.g. delivery vans)
        rng = np.random.default_rng(seed)
        self.vehicles: List[EVVehicle] = []
        for v in range(n_vehicles):
            cap = rng.choice([60.0, 75.0, 90.0]) # kWh
            max_p = rng.choice([11.0, 22.0]) # Level 2 AC vs fast AC charger kW
            # Staggered depot arrivals between 16:00 and 19:00
            arr = int(rng.integers(16, 20))
            # Departures between 06:00 and 08:00
            dep = int(rng.integers(6, 9))
            soc0 = float(rng.uniform(0.18, 0.35))
            self.vehicles.append(EVVehicle(v, cap, max_p, arr, dep, soc0, target_soc=0.90))

    def is_vehicle_plugged_in(self, v_idx: int, t: int) -> bool:
        """Determines if vehicle is at depot during hour t."""
        v = self.vehicles[v_idx]
        if v.arrival > v.departure:
            # Overnights: arrives late afternoon, departs next morning
            return (t >= v.arrival) or (t < v.departure)
        else:
            return v.arrival <= t < v.departure
