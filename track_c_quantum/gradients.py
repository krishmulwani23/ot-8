"""
Quantum Gradient Computation Methods (Track C):
1. Parameter-Shift Rule (exact analytical gradient on quantum hardware)
2. SPSA (Simultaneous Perturbation Stochastic Approximation - 2 evals per step)
3. Quantum Natural Gradient (QNG with Fubini-Study metric tensor)
"""

import numpy as np
from typing import Callable, Tuple, List, Dict, Any

class QuantumGradients:
    @staticmethod
    def parameter_shift_gradient(circuit_fn: Callable[[np.ndarray], float],
                                 params: np.ndarray, shift: float = np.pi / 2.0) -> np.ndarray:
        """
        Exact Parameter-Shift Rule:
        d<H>/d theta_i = ( <H>(theta_i + s) - <H>(theta_i - s) ) / (2 * sin(s))
        For s = pi/2: 2 * sin(pi/2) = 2.
        Evaluates 2 * p quantum circuits.
        """
        p = len(params)
        grad = np.zeros(p, dtype=np.float64)

        for i in range(p):
            p_pos = params.copy()
            p_neg = params.copy()
            p_pos[i] += shift
            p_neg[i] -= shift

            exp_pos = circuit_fn(p_pos)
            exp_neg = circuit_fn(p_neg)
            grad[i] = (exp_pos - exp_neg) / (2.0 * np.sin(shift))

        return grad

    @staticmethod
    def spsa_gradient(circuit_fn: Callable[[np.ndarray], float],
                      params: np.ndarray, c: float = 0.1,
                      rng: np.random.Generator = None) -> np.ndarray:
        """
        Simultaneous Perturbation Stochastic Approximation (SPSA):
        Requires only TWO circuit evaluations regardless of parameter dimension p!
        Delta_k drawn uniformly from {-1, +1}.
        """
        if rng is None:
            rng = np.random.default_rng()

        p = len(params)
        delta = rng.choice([-1.0, 1.0], size=p)

        theta_pos = params + c * delta
        theta_neg = params - c * delta

        exp_pos = circuit_fn(theta_pos)
        exp_neg = circuit_fn(theta_neg)

        diff = exp_pos - exp_neg
        grad = diff / (2.0 * c * delta)
        return grad

    @staticmethod
    def fubini_study_metric(state_fn: Callable[[np.ndarray], np.ndarray],
                            params: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Computes Fubini-Study Metric Tensor g_ij(theta):
        g_ij = Re( <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> )
        using finite-difference approximations of partial state derivatives.
        """
        p = len(params)
        psi0 = state_fn(params)
        d_psi = np.zeros((p, len(psi0)), dtype=np.complex128)

        for i in range(p):
            p_pos = params.copy()
            p_neg = params.copy()
            p_pos[i] += eps
            p_neg[i] -= eps
            psi_pos = state_fn(p_pos)
            psi_neg = state_fn(p_neg)
            d_psi[i] = (psi_pos - psi_neg) / (2.0 * eps)

        metric = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            for j in range(p):
                term1 = np.vdot(d_psi[i], d_psi[j])
                term2 = np.vdot(d_psi[i], psi0) * np.vdot(psi0, d_psi[j])
                metric[i, j] = np.real(term1 - term2)

        return metric

    @staticmethod
    def quantum_natural_gradient_step(grad: np.ndarray, metric: np.ndarray,
                                      reg: float = 1e-3) -> np.ndarray:
        """
        QNG step: delta = (G + reg * I)^{-1} grad
        Updates along the steepest descent direction on the Fubini-Study quantum manifold.
        """
        p = len(grad)
        G_reg = metric + reg * np.eye(p)
        try:
            step = np.linalg.solve(G_reg, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(G_reg) @ grad
        return step
