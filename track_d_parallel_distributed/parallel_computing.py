"""
Parallel Program Optimization (Track D):
- Static chunking vs Dynamic work stealing under load imbalance
- Synchronization and communication overhead
- Speedup and parallel efficiency scaling across thread counts
- Amdahl's Law and Gustafson's Law model fitting and interpretation
"""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Tuple, Callable, Optional
import queue
import threading

def synthetic_task(complexity: float) -> float:
    """Workload simulating irregular compute tasks (matrix operations & transcendental math)."""
    # Deterministic compute loop proportional to complexity
    res = 0.0
    n = int(max(10, complexity * 8000))
    for i in range(n):
        res += np.sin(i * 0.001) * np.cos(i * 0.002)
    return res


class WorkStealingQueue:
    """Simulates a per-worker work-stealing double-ended queue."""
    def __init__(self, tasks: List[float]):
        self.tasks = list(tasks)
        self.lock = threading.Lock()

    def pop_local(self) -> Optional[float]:
        """Local worker pops from tail (LIFO for cache locality)."""
        with self.lock:
            return self.tasks.pop() if self.tasks else None

    def steal(self) -> Optional[float]:
        """Foreign worker steals from head (FIFO for load balancing)."""
        with self.lock:
            return self.tasks.pop(0) if self.tasks else None


class ParallelBenchmarking:
    @staticmethod
    def generate_irregular_workload(n_tasks: int = 400, seed: int = 42) -> np.ndarray:
        """Generates tasks with heavy-tailed / Pareto complexity distribution (creates load imbalance)."""
        rng = np.random.default_rng(seed)
        # Pareto distribution: 80% tasks are light, 20% tasks are extremely heavy
        complexities = rng.pareto(a=1.5, size=n_tasks) * 0.5 + 0.1
        return np.clip(complexities, 0.1, 5.0)

    @staticmethod
    def static_chunking_execution(tasks: np.ndarray, n_workers: int) -> Tuple[float, List[float]]:
        """Static chunking: partitions tasks equally into n_workers contiguous blocks upfront."""
        chunks = np.array_split(tasks, n_workers)
        worker_times = [0.0] * n_workers

        def worker_fn(worker_id: int):
            t_start = time.perf_counter()
            for c in chunks[worker_id]:
                synthetic_task(c)
            worker_times[worker_id] = (time.perf_counter() - t_start) * 1000.0

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker_fn, args=(i,)) for i in range(n_workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        return total_time_ms, worker_times

    @staticmethod
    def work_stealing_execution(tasks: np.ndarray, n_workers: int) -> Tuple[float, List[float]]:
        """Dynamic work-stealing execution: idle workers steal tasks from busiest queues."""
        chunks = np.array_split(tasks, n_workers)
        queues = [WorkStealingQueue(c.tolist()) for c in chunks]
        worker_times = [0.0] * n_workers

        def worker_fn(worker_id: int):
            t_start = time.perf_counter()
            my_q = queues[worker_id]
            while True:
                task = my_q.pop_local()
                if task is not None:
                    synthetic_task(task)
                    continue

                # Try to steal from other workers
                stolen_task = None
                for victim_id in range(n_workers):
                    if victim_id != worker_id:
                        stolen_task = queues[victim_id].steal()
                        if stolen_task is not None:
                            break

                if stolen_task is not None:
                    synthetic_task(stolen_task)
                else:
                    # Check if all queues are genuinely empty
                    all_empty = all(len(q.tasks) == 0 for q in queues)
                    if all_empty:
                        break
                    time.sleep(0.0001)

            worker_times[worker_id] = (time.perf_counter() - t_start) * 1000.0

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker_fn, args=(i,)) for i in range(n_workers)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        total_time_ms = (time.perf_counter() - t0) * 1000.0
        return total_time_ms, worker_times

    @staticmethod
    def benchmark_scaling(tasks: np.ndarray, worker_counts: List[int] = [1, 2, 4, 8]) -> Dict[str, Any]:
        """Measures execution times, speedup, and efficiency across varying worker counts."""
        # 1. Baseline: 1 worker
        t1_static, _ = ParallelBenchmarking.static_chunking_execution(tasks, 1)

        static_times = []
        static_speedups = []
        static_efficiencies = []

        stealing_times = []
        stealing_speedups = []
        stealing_efficiencies = []

        for w in worker_counts:
            # Static chunking
            t_stat, _ = ParallelBenchmarking.static_chunking_execution(tasks, w)
            sp_stat = t1_static / t_stat
            eff_stat = sp_stat / w
            static_times.append(t_stat)
            static_speedups.append(sp_stat)
            static_efficiencies.append(eff_stat)

            # Work stealing
            t_steal, _ = ParallelBenchmarking.work_stealing_execution(tasks, w)
            sp_steal = t1_static / t_steal
            eff_steal = sp_steal / w
            stealing_times.append(t_steal)
            stealing_speedups.append(sp_steal)
            stealing_efficiencies.append(eff_steal)

        # Fit Amdahl's Law: S(p) = 1 / ((1 - f) + f / p)
        # f is parallel fraction: fit using non-linear least squares / grid search
        best_f = 0.95
        best_err = float("inf")
        for f_cand in np.linspace(0.5, 0.999, 500):
            preds = [1.0 / ((1.0 - f_cand) + f_cand / p) for p in worker_counts]
            err = np.sum((np.array(preds) - np.array(stealing_speedups))**2)
            if err < best_err:
                best_err = err
                best_f = f_cand

        return {
            "worker_counts": worker_counts,
            "t1_baseline_ms": t1_static,
            "static_chunking": {
                "runtimes_ms": static_times,
                "speedup": static_speedups,
                "efficiency": static_efficiencies
            },
            "work_stealing": {
                "runtimes_ms": stealing_times,
                "speedup": stealing_speedups,
                "efficiency": stealing_efficiencies
            },
            "amdahl_fit": {
                "estimated_parallel_fraction_f": float(best_f),
                "theoretical_max_speedup": float(1.0 / (1.0 - best_f))
            }
        }
