"""
QUBO and Ising Formulations (Track B & baseline for Track C):
- Binary-to-spin variable transformation: x_i in {0, 1} <-> s_i in {-1, +1}
- Penalty terms for constraints
- Max-Cut graph problem formulation
- Classical QUBO Solvers: Simulated Annealing, Exact Brute-Force, and ILP baseline
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.optimize import milp, LinearConstraint, Bounds

class QUBOInstance:
    """
    Represents Quadratic Unconstrained Binary Optimization:
    Minimize: E(x) = x^T Q x
    where x in {0, 1}^n and Q is upper triangular or symmetric matrix.
    """
    def __init__(self, Q: np.ndarray):
        self.Q = np.asarray(Q, dtype=np.float64)
        self.n = self.Q.shape[0]
        assert self.Q.shape == (self.n, self.n), "Q must be square matrix"

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(x @ self.Q @ x)

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Converts QUBO to Ising spin Hamiltonian:
        x_i = (1 + s_i) / 2 with s_i in {-1, +1}
        Returns:
            J: Coupling matrix (J_ij for i < j)
            h: Longitudinal magnetic field vector
            offset: Energy constant scalar
        """
        n = self.n
        Q_sym = 0.5 * (self.Q + self.Q.T)
        J = np.zeros((n, n), dtype=np.float64)
        h = np.zeros(n, dtype=np.float64)
        offset = 0.0

        for i in range(n):
            for j in range(n):
                q = Q_sym[i, j]
                # x_i * x_j = (1 + s_i)(1 + s_j) / 4 = 1/4 + s_i/4 + s_j/4 + s_i s_j / 4
                if i != j:
                    J[i, j] += q / 4.0
                    h[i] += q / 4.0
                    h[j] += q / 4.0
                    offset += q / 4.0
                else:
                    # x_i^2 = x_i = (1 + s_i) / 2
                    h[i] += q / 2.0
                    offset += q / 2.0

        return J, h, float(offset)

    @staticmethod
    def ising_to_qubo(J: np.ndarray, h: np.ndarray, offset: float) -> 'QUBOInstance':
        """
        Converts Ising Hamiltonian (s in {-1, +1}^n) to QUBO:
        s_i = 2*x_i - 1 with x_i in {0, 1}
        """
        n = len(h)
        Q = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    Q[i, j] += 4.0 * J[i, j]
            Q[i, i] += 2.0 * h[i] - 4.0 * np.sum(J[i, :])
        return QUBOInstance(Q)


class MaxCutProblem:
    r"""
    Max-Cut Problem on Graph G=(V, E):
    Partition vertices into S and V\S to maximize edges crossing between S and V\S.
    Cut(x) = sum_{(u, v) in E} (x_u + x_v - 2*x_u*x_v)
    Minimization QUBO: -Cut(x).
    Ising model: Minimize sum_{(u, v) in E} s_u * s_v.
    """
    def __init__(self, n_vertices: int, edges: List[Tuple[int, int, float]]):
        self.n = n_vertices
        self.edges = edges # list of (u, v, weight)
        self.adj = np.zeros((self.n, self.n), dtype=np.float64)
        for u, v, w in edges:
            self.adj[u, v] += w
            self.adj[v, u] += w

    def to_qubo(self) -> QUBOInstance:
        """Constructs QUBO matrix Q such that minimizing x^T Q x maximizes Cut."""
        Q = np.zeros((self.n, self.n), dtype=np.float64)
        for u, v, w in self.edges:
            # -w * (x_u + x_v - 2*x_u*x_v) = -w*x_u - w*x_v + 2*w*x_u*x_v
            Q[u, u] -= w
            Q[v, v] -= w
            Q[u, v] += w
            Q[v, u] += w
        return QUBOInstance(Q)

    def calculate_cut(self, x: np.ndarray) -> float:
        cut = 0.0
        for u, v, w in self.edges:
            if x[u] != x[v]:
                cut += w
        return float(cut)


class ClassicalQUBOSolvers:
    """Classical benchmark solvers for QUBO / Ising problems."""

    @staticmethod
    def brute_force(qubo: QUBOInstance) -> Dict[str, Any]:
        """Exact enumeration over all 2^n binary states (for n <= 20)."""
        t0 = time.perf_counter()
        n = qubo.n
        best_val = float("inf")
        best_x = np.zeros(n, dtype=int)

        for bit in range(1 << n):
            x = np.array([(bit >> k) & 1 for k in range(n)], dtype=np.float64)
            val = qubo.evaluate(x)
            if val < best_val:
                best_val = val
                best_x = x.astype(int)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "solver": "Exact_Brute_Force",
            "best_energy": float(best_val),
            "best_x": best_x.tolist(),
            "runtime_ms": elapsed_ms
        }

    @staticmethod
    def simulated_annealing(qubo: QUBOInstance, n_steps: int = 5000,
                            temp_start: float = 10.0, temp_end: float = 0.01,
                            seed: int = 42) -> Dict[str, Any]:
        """Metropolis-Hastings Simulated Annealing algorithm."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = qubo.n
        x = rng.integers(0, 2, size=n).astype(np.float64)
        current_energy = qubo.evaluate(x)
        best_energy = current_energy
        best_x = x.copy()

        decay = (temp_end / temp_start) ** (1.0 / n_steps)
        T = temp_start

        for step in range(n_steps):
            idx = rng.integers(0, n)
            x_flip = x.copy()
            x_flip[idx] = 1.0 - x_flip[idx]
            new_energy = qubo.evaluate(x_flip)
            delta_e = new_energy - current_energy

            if delta_e < 0 or rng.uniform(0, 1) < np.exp(-delta_e / max(1e-9, T)):
                x = x_flip
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()

            T *= decay

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "solver": "Simulated_Annealing",
            "best_energy": float(best_energy),
            "best_x": best_x.astype(int).tolist(),
            "runtime_ms": elapsed_ms
        }
