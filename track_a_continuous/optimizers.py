"""
First-order optimization algorithms for Continuous Optimization (Track A):
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Mini-batch SGD
- Momentum
- Nesterov Accelerated Gradient (NAG)
- AdaGrad
- RMSProp
- Adam
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional

class OptimizerState:
    def __init__(self, x: np.ndarray):
        self.x = np.array(x, dtype=np.float64, copy=True)
        self.trajectory: List[np.ndarray] = [self.x.copy()]
        self.losses: List[float] = []
        self.grad_norms: List[float] = []
        self.iterations: int = 0
        self.converged: bool = False


class BaseOptimizer:
    def __init__(self, lr: float = 0.01, max_iter: int = 500, tol: float = 1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.name = self.__class__.__name__

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        pass

    def optimize(self, func: Callable[[np.ndarray], float],
                 grad_fn: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray) -> OptimizerState:
        self.reset()
        state = OptimizerState(x0)
        state.losses.append(func(state.x))
        state.grad_norms.append(float(np.linalg.norm(grad_fn(state.x))))

        for it in range(self.max_iter):
            g = grad_fn(state.x)
            gnorm = float(np.linalg.norm(g))
            if gnorm < self.tol:
                state.converged = True
                break

            state.x = self.step(state.x, g)
            state.trajectory.append(state.x.copy())
            loss = func(state.x)
            state.losses.append(loss)
            state.grad_norms.append(gnorm)
            state.iterations += 1

            if not np.isfinite(loss):
                break

        return state


class GradientDescent(BaseOptimizer):
    """Standard Gradient Descent: x_{t+1} = x_t - lr * grad"""
    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return x - self.lr * grad


class MomentumOptimizer(BaseOptimizer):
    """
    Polyak Classical Momentum:
    v_{t+1} = beta * v_t + lr * grad
    x_{t+1} = x_t - v_{t+1}
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9, max_iter: int = 500, tol: float = 1e-6):
        super().__init__(lr, max_iter, tol)
        self.beta = beta
        self.v: Optional[np.ndarray] = None

    def reset(self):
        self.v = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v + self.lr * grad
        return x - self.v


class NesterovMomentum(BaseOptimizer):
    """
    Nesterov Accelerated Gradient (NAG):
    Lookahead point: x_lookahead = x_t - beta * v_t
    v_{t+1} = beta * v_t + lr * grad(x_lookahead)
    x_{t+1} = x_t - v_{t+1}
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9, max_iter: int = 500, tol: float = 1e-6):
        super().__init__(lr, max_iter, tol)
        self.beta = beta
        self.v: Optional[np.ndarray] = None

    def reset(self):
        self.v = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        # Note: in standard form when grad is passed at current x:
        # v_{t+1} = beta * v_t + lr * grad
        # x_{t+1} = x_t - beta * v_{t+1} - lr * grad
        if self.v is None:
            self.v = np.zeros_like(x)
        v_prev = self.v.copy()
        self.v = self.beta * self.v + self.lr * grad
        return x - (self.beta * self.v + self.lr * grad)


class AdaGrad(BaseOptimizer):
    """
    AdaGrad:
    G_{t+1} = G_t + grad^2
    x_{t+1} = x_t - (lr / sqrt(G_{t+1} + eps)) * grad
    """
    def __init__(self, lr: float = 0.1, eps: float = 1e-8, max_iter: int = 500, tol: float = 1e-6):
        super().__init__(lr, max_iter, tol)
        self.eps = eps
        self.G: Optional[np.ndarray] = None

    def reset(self):
        self.G = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros_like(x)
        self.G += grad**2
        return x - (self.lr / (np.sqrt(self.G) + self.eps)) * grad


class RMSProp(BaseOptimizer):
    """
    RMSProp:
    v_{t+1} = beta * v_t + (1 - beta) * grad^2
    x_{t+1} = x_t - (lr / sqrt(v_{t+1} + eps)) * grad
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8,
                 max_iter: int = 500, tol: float = 1e-6):
        super().__init__(lr, max_iter, tol)
        self.beta = beta
        self.eps = eps
        self.v: Optional[np.ndarray] = None

    def reset(self):
        self.v = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(x)
        self.v = self.beta * self.v + (1.0 - self.beta) * (grad**2)
        return x - (self.lr / (np.sqrt(self.v) + self.eps)) * grad


class Adam(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation):
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    x_t = x_{t-1} - (lr / (sqrt(v_hat) + eps)) * m_hat
    """
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, max_iter: int = 500, tol: float = 1e-6):
        super().__init__(lr, max_iter, tol)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)

        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)

        return x - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


class MiniBatchSGDOptimizer:
    """
    Mini-batch SGD implementation on dataset samples (X, y) with loss and loss gradient.
    Demonstrates SGD (batch_size=1), Mini-batch SGD (1 < batch_size < N), and full-batch GD (batch_size=N).
    """
    def __init__(self, lr: float = 0.05, batch_size: int = 16, epochs: int = 50, seed: int = 42):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray,
            loss_and_grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, np.ndarray]],
            w0: np.ndarray) -> Dict[str, Any]:
        N = X.shape[0]
        w = np.array(w0, dtype=np.float64, copy=True)
        history_loss = []
        trajectory = [w.copy()]

        for epoch in range(self.epochs):
            indices = self.rng.permutation(N)
            epoch_loss = 0.0
            num_batches = 0

            for start_idx in range(0, N, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                xb = X[batch_idx]
                yb = y[batch_idx]

                b_loss, b_grad = loss_and_grad_fn(w, xb, yb)
                w -= self.lr * b_grad
                epoch_loss += b_loss
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            history_loss.append(avg_loss)
            trajectory.append(w.copy())

        return {
            "w_final": w,
            "losses": history_loss,
            "trajectory": trajectory,
            "batch_size": self.batch_size
        }
