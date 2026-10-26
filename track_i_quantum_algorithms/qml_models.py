"""
Quantum Machine Learning (QML) vs Classical Models (Track I):
- Variational Quantum Classifier (VQC) with Parameterized Quantum Circuit
- Classical Multi-Layer Perceptron (MLP) & Linear Classifier
- Evaluated on identical non-linear classification dataset.
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, List
from track_c_quantum.quantum_engine import QuantumCircuit
from track_c_quantum.gradients import QuantumGradients

class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier (VQC) on 2 qubits:
    Feature map: S(x) = Ry(x_0) on q0, Ry(x_1) on q1
    Ansatz: Ry(theta_0), Rz(theta_1) on q0; Ry(theta_2), Rz(theta_3) on q1; CX(0, 1)
            Ry(theta_4) on q0; Ry(theta_5) on q1
    Observable: Pauli Z on qubit 0
    Prediction: y_hat = 1 if <Z0> > 0 else 0
    """
    def __init__(self, n_params: int = 6, seed: int = 42):
        self.n_params = n_params
        self.rng = np.random.default_rng(seed)
        self.theta = self.rng.uniform(0, 2*np.pi, size=n_params)
        self.total_circuit_evals = 0

    def _forward_circuit(self, x: np.ndarray, theta: np.ndarray) -> float:
        self.total_circuit_evals += 1
        qc = QuantumCircuit(2)
        # 1. Feature Encoding
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)

        # 2. Parameterized Variational Ansatz
        qc.ry(theta[0], 0)
        qc.rz(theta[1], 0)
        qc.ry(theta[2], 1)
        qc.rz(theta[3], 1)
        qc.cx(0, 1)
        qc.ry(theta[4], 0)
        qc.ry(theta[5], 1)

        # 3. Expectation of Z on qubit 0: <Z0> = (P(00) + P(01)) - (P(10) + P(11))
        # In computational basis: qubit 0 is 0 for idx in {0, 2}, 1 for idx in {1, 3}
        probs = qc.probabilities()
        diag_z0 = np.array([1.0, -1.0, 1.0, -1.0])
        return float(np.sum(probs * diag_z0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            exp_z = self._forward_circuit(x, self.theta)
            preds.append(1 if exp_z >= 0.0 else 0)
        return np.array(preds)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 15, lr: float = 0.1) -> Dict[str, Any]:
        t0 = time.perf_counter()
        N = len(y)
        # Convert y in {0, 1} to {-1, +1} target
        y_target = 2.0 * y - 1.0
        losses = []

        for ep in range(epochs):
            total_loss = 0.0
            grad_accum = np.zeros(self.n_params)

            # Mini-batch or sample gradients
            sample_indices = self.rng.choice(N, size=min(16, N), replace=False)
            for idx in sample_indices:
                x_i = X[idx]
                y_i = y_target[idx]

                def loss_sample(th):
                    pred_i = self._forward_circuit(x_i, th)
                    return (pred_i - y_i)**2

                loss_val = loss_sample(self.theta)
                total_loss += loss_val

                # Parameter shift gradient on sample loss
                g = QuantumGradients.parameter_shift_gradient(loss_sample, self.theta)
                grad_accum += g

            grad_accum /= len(sample_indices)
            self.theta -= lr * grad_accum
            losses.append(total_loss / len(sample_indices))

        train_time_ms = (time.perf_counter() - t0) * 1000.0
        preds = self.predict(X)
        acc = float(np.mean(preds == y)) * 100.0

        return {
            "model": "Variational_Quantum_Classifier",
            "accuracy_pct": acc,
            "train_time_ms": train_time_ms,
            "parameter_count": self.n_params,
            "circuit_evaluations": self.total_circuit_evals,
            "final_loss": losses[-1] if losses else 0.0
        }


class ClassicalClassifier:
    """Equivalent Classical 2-Layer Perceptron on identical dataset."""
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.normal(0, 0.5, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0, 0.5, (hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.n_params = self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def predict(self, X: np.ndarray) -> np.ndarray:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        probs = 1.0 / (1.0 + np.exp(-np.clip(z2, -20, 20)))
        return (probs.ravel() >= 0.5).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 80, lr: float = 0.1) -> Dict[str, Any]:
        t0 = time.perf_counter()
        N = len(y)
        y_col = y.reshape(-1, 1)

        for ep in range(epochs):
            z1 = X @ self.W1 + self.b1
            a1 = np.maximum(0, z1)
            z2 = a1 @ self.W2 + self.b2
            y_hat = 1.0 / (1.0 + np.exp(-np.clip(z2, -20, 20)))

            dz2 = (y_hat - y_col) / N
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)

            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        train_time_ms = (time.perf_counter() - t0) * 1000.0
        preds = self.predict(X)
        acc = float(np.mean(preds == y)) * 100.0

        return {
            "model": "Classical_MLP",
            "accuracy_pct": acc,
            "train_time_ms": train_time_ms,
            "parameter_count": self.n_params,
            "circuit_evaluations": 0
        }
