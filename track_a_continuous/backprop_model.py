"""
Neural Network demonstrating Backpropagation (gradient computation)
strictly decoupled from the Optimizer (parameter update rule).
Architecture: 2-layer MLP (Input -> Linear -> ReLU -> Linear -> Sigmoid/MSE or CrossEntropy)
"""

import numpy as np
from typing import Dict, Tuple, List

class TwoLayerMLP:
    """
    Two-layer Multi-Layer Perceptron:
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    Loss = Binary Cross Entropy (BCE) with L2 regularization
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 reg_lambda: float = 1e-4, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda
        self.rng = np.random.default_rng(seed)

        # He initialization
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_flat_params(self, theta: np.ndarray):
        idx = 0
        w1_size = self.W1.size
        self.W1 = theta[idx:idx + w1_size].reshape(self.W1.shape)
        idx += w1_size

        b1_size = self.b1.size
        self.b1 = theta[idx:idx + b1_size].reshape(self.b1.shape)
        idx += b1_size

        w2_size = self.W2.size
        self.W2 = theta[idx:idx + w2_size].reshape(self.W2.shape)
        idx += w2_size

        b2_size = self.b2.size
        self.b2 = theta[idx:idx + b2_size].reshape(self.b2.shape)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z_clamped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clamped))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward pass caching intermediate activations."""
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        y_hat = self.sigmoid(z2)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}
        return y_hat, cache

    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """BCE loss with L2 weight regularization."""
        eps = 1e-12
        y_hat_c = np.clip(y_hat, eps, 1.0 - eps)
        bce = -np.mean(y * np.log(y_hat_c) + (1.0 - y) * np.log(1.0 - y_hat_c))
        l2 = 0.5 * self.reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return float(bce + l2)

    def backpropagation(self, y: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Backpropagation (Reverse-Mode Automatic Differentiation):
        Computes exact partial derivatives dL/dW1, dL/db1, dL/dW2, dL/db2.
        Notice: this step computes the gradients; it does NOT update any weights!
        """
        X = cache["X"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        y_hat = cache["y_hat"]
        N = X.shape[0]

        # dL/dz2 = (y_hat - y) / N for BCE + sigmoid
        dz2 = (y_hat - y) / float(N)

        # Gradients for layer 2
        dW2 = a1.T @ dz2 + self.reg_lambda * self.W2
        db2 = np.sum(dz2, axis=0)

        # Backprop through ReLU
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)

        # Gradients for layer 1
        dW1 = X.T @ dz1 + self.reg_lambda * self.W1
        db1 = np.sum(dz1, axis=0)

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
            "flat_grad": np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
        }

    def compute_loss_and_flat_grad(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Helper for optimization routines operating on flat parameter vector."""
        self.set_flat_params(theta)
        y_hat, cache = self.forward(X)
        l = self.loss(y_hat, y)
        grads = self.backpropagation(y, cache)
        return l, grads["flat_grad"]

    def verify_gradients_numerical(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
        """Finite difference check validating backpropagation exactness."""
        theta = self.get_flat_params()
        _, ana_grad = self.compute_loss_and_flat_grad(theta, X, y)
        num_grad = np.zeros_like(theta)

        for i in range(min(50, len(theta))): # check first 50 coordinates
            theta_pos = theta.copy()
            theta_neg = theta.copy()
            theta_pos[i] += eps
            theta_neg[i] -= eps
            l_pos, _ = self.compute_loss_and_flat_grad(theta_pos, X, y)
            l_neg, _ = self.compute_loss_and_flat_grad(theta_neg, X, y)
            num_grad[i] = (l_pos - l_neg) / (2.0 * eps)

        diff = np.linalg.norm(ana_grad[:50] - num_grad[:50]) / (np.linalg.norm(ana_grad[:50]) + np.linalg.norm(num_grad[:50]) + 1e-12)
        return float(diff)
