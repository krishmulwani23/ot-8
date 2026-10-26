"""
Objective functions and loss landscapes for Continuous Optimization (Track A).
Includes:
- Strongly convex quadratic (ill-conditioned)
- Non-convex with saddle points (Hyperbolic paraboloid, Monkey saddle)
- Non-convex with local minima (Rosenbrock, Rastrigin)
- Noisy objective wrapper
"""

import numpy as np

class ObjectiveFunction:
    """Base class for test objective functions with gradient and Hessian."""
    def __init__(self, name: str, is_convex: bool, global_optimum: np.ndarray, optimal_value: float):
        self.name = name
        self.is_convex = is_convex
        self.global_optimum = np.array(global_optimum, dtype=np.float64)
        self.optimal_value = optimal_value

    def value(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hessian_vector_product(self, x: np.ndarray, v: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Hessian-vector product using two-point central difference of gradients."""
        grad_pos = self.gradient(x + eps * v)
        grad_neg = self.gradient(x - eps * v)
        return (grad_pos - grad_neg) / (2.0 * eps)


class ConvexQuadratic(ObjectiveFunction):
    """
    Strongly convex quadratic: f(x, y) = 0.5 * (x^2 + kappa * y^2)
    Condition number kappa controls landscape anisotropy.
    """
    def __init__(self, kappa: float = 10.0):
        super().__init__("Convex Quadratic (kappa=%.1f)" % kappa, is_convex=True,
                         global_optimum=np.array([0.0, 0.0]), optimal_value=0.0)
        self.kappa = float(kappa)
        self.Q = np.diag([1.0, self.kappa])

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return 0.5 * float(x @ self.Q @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.Q @ x

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.Q.copy()


class SaddlePointLandscape(ObjectiveFunction):
    """
    Hyperbolic Paraboloid: f(x, y) = x^2 - y^2
    Saddle point at (0, 0) where Hessian has eigenvalues +2 and -2.
    """
    def __init__(self):
        super().__init__("Hyperbolic Saddle Point", is_convex=False,
                         global_optimum=np.array([0.0, 0.0]), optimal_value=0.0)

    def value(self, x: np.ndarray) -> float:
        return float(x[0]**2 - x[1]**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * x[0], -2.0 * x[1]], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2.0, 0.0], [0.0, -2.0]], dtype=np.float64)


class Rosenbrock(ObjectiveFunction):
    """
    Rosenbrock Valley: f(x, y) = (a - x)^2 + b*(y - x^2)^2
    Standard parameters: a=1, b=100.
    Narrow curved parabolic valley with global minimum at (1, 1).
    """
    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__("Rosenbrock Valley", is_convex=False,
                         global_optimum=np.array([a, a**2]), optimal_value=0.0)
        self.a = a
        self.b = b

    def value(self, x: np.ndarray) -> float:
        return float((self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        dx = -2.0 * (self.a - x[0]) - 4.0 * self.b * x[0] * (x[1] - x[0]**2)
        dy = 2.0 * self.b * (x[1] - x[0]**2)
        return np.array([dx, dy], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        dxx = 2.0 - 4.0 * self.b * (x[1] - 3.0 * x[0]**2)
        dxy = -4.0 * self.b * x[0]
        dyy = 2.0 * self.b
        return np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float64)


class Rastrigin(ObjectiveFunction):
    """
    Rastrigin Function: highly multimodal non-convex with many local minima.
    f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum at x = 0 with f(x) = 0.
    """
    def __init__(self, dim: int = 2):
        super().__init__("Rastrigin (%dd)" % dim, is_convex=False,
                         global_optimum=np.zeros(dim), optimal_value=0.0)
        self.dim = dim

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(10.0 * self.dim + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        diag = 2.0 + 40.0 * (np.pi**2) * np.cos(2.0 * np.pi * x)
        return np.diag(diag)


class NoisyObjective:
    """Wrapper that injects Gaussian noise into gradient evaluations (simulates stochastic noise)."""
    def __init__(self, base_objective: ObjectiveFunction, noise_std: float = 0.1, seed: int = 42):
        self.base = base_objective
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.name = f"{base_objective.name} (Noisy sigma={noise_std})"
        self.is_convex = base_objective.is_convex
        self.global_optimum = base_objective.global_optimum
        self.optimal_value = base_objective.optimal_value

    def value(self, x: np.ndarray) -> float:
        return self.base.value(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        true_grad = self.base.gradient(x)
        noise = self.rng.normal(0.0, self.noise_std, size=true_grad.shape)
        return true_grad + noise

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.base.hessian(x)
