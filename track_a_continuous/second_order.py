"""
Second-order and curvature-based optimization methods (Track A):
- Newton-Raphson (with Levenberg-Marquardt regularization)
- Gauss-Newton for nonlinear least squares
- Curvature approximations:
  * Empirical Fisher Information Matrix
  * Diagonal and Low-Rank (BFGS) Hessian approximation
  * Hessian-Vector Products (HVP) and Newton-CG (Truncated Newton)
"""

import numpy as np
from typing import Callable, Tuple, List, Dict, Any, Optional

class NewtonOptimizer:
    """
    Classical Newton-Raphson method:
    x_{t+1} = x_t - [H(x_t) + reg * I]^{-1} grad(x_t)
    Regularization ensures positive definiteness even near saddle points.
    """
    def __init__(self, lr: float = 1.0, reg: float = 1e-4, max_iter: int = 100, tol: float = 1e-6):
        self.lr = lr
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.name = "Newton-Raphson"

    def optimize(self, func: Callable[[np.ndarray], float],
                 grad_fn: Callable[[np.ndarray], np.ndarray],
                 hess_fn: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray) -> Dict[str, Any]:
        x = np.array(x0, dtype=np.float64, copy=True)
        trajectory = [x.copy()]
        losses = [func(x)]
        grad_norms = [float(np.linalg.norm(grad_fn(x)))]

        for it in range(self.max_iter):
            g = grad_fn(x)
            gnorm = float(np.linalg.norm(g))
            if gnorm < self.tol:
                break

            H = hess_fn(x)
            # Regularize Hessian to handle non-positive definite cases (saddles)
            H_reg = H + self.reg * np.eye(len(x))
            try:
                delta = np.linalg.solve(H_reg, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H_reg) @ g

            x = x - self.lr * delta
            trajectory.append(x.copy())
            loss = func(x)
            losses.append(loss)
            grad_norms.append(gnorm)

        return {
            "x_final": x,
            "losses": losses,
            "trajectory": trajectory,
            "grad_norms": grad_norms,
            "iterations": len(losses) - 1
        }


class GaussNewtonOptimizer:
    """
    Gauss-Newton method for Non-linear Least Squares:
    Objective: f(x) = 0.5 * || r(x) ||^2
    Jacobian: J = dr/dx
    Hessian approx: H_GN = J^T J
    Step: (J^T J + damping * I) delta = J^T r
    """
    def __init__(self, damping: float = 1e-3, max_iter: int = 100, tol: float = 1e-6):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.name = "Gauss-Newton"

    def optimize(self, residual_fn: Callable[[np.ndarray], np.ndarray],
                 jacobian_fn: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray) -> Dict[str, Any]:
        x = np.array(x0, dtype=np.float64, copy=True)
        trajectory = [x.copy()]
        r0 = residual_fn(x)
        losses = [0.5 * float(np.sum(r0**2))]

        for it in range(self.max_iter):
            r = residual_fn(x)
            J = jacobian_fn(x)
            grad = J.T @ r
            gnorm = float(np.linalg.norm(grad))
            if gnorm < self.tol:
                break

            H_approx = J.T @ J + self.damping * np.eye(len(x))
            try:
                delta = np.linalg.solve(H_approx, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H_approx) @ grad

            x = x - delta
            trajectory.append(x.copy())
            losses.append(0.5 * float(np.sum(residual_fn(x)**2)))

        return {
            "x_final": x,
            "losses": losses,
            "trajectory": trajectory,
            "iterations": len(losses) - 1
        }


class CurvatureApproximations:
    """
    Tools demonstrating:
    1. Fisher Information Matrix (empirical Fisher)
    2. Diagonal and Low-Rank (BFGS) curvature approximations
    3. Hessian-Vector Products (HVP) and Newton-CG
    """

    @staticmethod
    def empirical_fisher(sample_gradients: np.ndarray) -> np.ndarray:
        """
        Calculates the empirical Fisher Information Matrix F = (1/N) * sum_i g_i g_i^T
        sample_gradients: shape (N_samples, n_params)
        """
        N, d = sample_gradients.shape
        F = np.zeros((d, d), dtype=np.float64)
        for i in range(N):
            g = sample_gradients[i:i+1] # (1, d)
            F += g.T @ g
        return F / float(N)

    @staticmethod
    def diagonal_hessian(H: np.ndarray) -> np.ndarray:
        """Extract diagonal curvature: O(d) storage instead of O(d^2)."""
        return np.diag(np.diag(H))

    @staticmethod
    def bfgs_update(B: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Low-rank Quasi-Newton BFGS update of inverse Hessian approximation B (or Hessian H):
        s = x_{t+1} - x_t
        y = g_{t+1} - g_t
        B_{t+1} = B - (B s s^T B)/(s^T B s) + (y y^T)/(y^T s)
        """
        sy = float(s @ y)
        if abs(sy) < 1e-12:
            return B
        Bs = B @ s
        sBs = float(s @ Bs)
        term1 = np.outer(Bs, Bs) / (sBs + 1e-12)
        term2 = np.outer(y, y) / sy
        return B - term1 + term2

    @staticmethod
    def newton_cg_step(grad: np.ndarray, hvp_fn: Callable[[np.ndarray], np.ndarray],
                       cg_max_iter: int = 20, tol: float = 1e-4) -> np.ndarray:
        """
        Hessian-free Truncated Newton step using Conjugate Gradient to solve H delta = grad.
        Uses ONLY Hessian-Vector Products: never forms the full matrix H!
        """
        d = len(grad)
        delta = np.zeros(d, dtype=np.float64)
        r = grad.copy() # residual: r = grad - H delta
        p = -r.copy()
        rs_old = float(r @ r)

        for _ in range(min(cg_max_iter, d)):
            if np.sqrt(rs_old) < tol:
                break
            Hp = hvp_fn(p)
            pHp = float(p @ Hp)
            if pHp <= 1e-12:
                # Negative or zero curvature encountered: exit in direction of p or current delta
                if np.linalg.norm(delta) == 0:
                    delta = -grad
                break
            alpha = rs_old / pHp
            delta += alpha * p
            r += alpha * Hp
            rs_new = float(r @ r)
            beta = rs_new / rs_old
            p = -r + beta * p
            rs_old = rs_new

        return delta
