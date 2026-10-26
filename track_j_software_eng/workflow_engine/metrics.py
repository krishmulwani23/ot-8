"""
Workflow Engine - Telemetry and Percentile Latency Metrics (Track J).
Calculates P50, P90, P99 execution metrics and throughput counters.
"""

import numpy as np
import threading
from typing import List, Dict, Any

class MetricsCollector:
    def __init__(self):
        self._latencies: List[float] = []
        self._counter: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record_latency(self, latency_ms: float):
        with self._lock:
            self._latencies.append(latency_ms)

    def increment_counter(self, name: str, amount: int = 1):
        with self._lock:
            self._counter[name] = self._counter.get(name, 0) + amount

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self._latencies:
                return {"count": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}
            arr = np.array(self._latencies)
            return {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
                "counters": dict(self._counter)
            }
