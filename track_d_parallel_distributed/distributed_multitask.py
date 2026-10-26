"""
Distributed and Multi-Task Optimization (Track D):
1. Federated Learning (FedAvg) vs Centralized Training across local steps E in {1, 5, 20}
2. Multi-Task Learning:
   - Equal Task Weighting baseline (w1 = w2 = 0.5)
   - Multiple Gradient Descent Algorithm (MGDA - Pareto optimality)
   - GradNorm (gradient magnitude normalization)
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class FederatedLearningFedAvg:
    """
    Federated Learning simulation using FedAvg (McMahan et al. 2017).
    Compares against centralized training.
    """
    def __init__(self, n_clients: int = 8, d_features: int = 10, seed: int = 42):
        self.n_clients = n_clients
        self.d = d_features
        self.rng = np.random.default_rng(seed)

        # True underlying ground truth parameter
        self.w_star = self.rng.normal(0, 1, self.d)

        # Distribute heterogeneous (non-IID) datasets among clients
        self.client_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.total_samples = 0
        for i in range(n_clients):
            n_i = self.rng.integers(30, 80)
            self.total_samples += n_i
            # Client bias shift creating non-IID heterogeneity
            shift = self.rng.normal(0, 0.5, self.d)
            Xi = self.rng.normal(0, 1, (n_i, self.d)) + shift
            yi = Xi @ self.w_star + self.rng.normal(0, 0.1, n_i)
            self.client_data.append((Xi, yi))

    def evaluate_global_loss(self, w: np.ndarray) -> float:
        """Computes average Mean Squared Error across all client datasets."""
        total_loss = 0.0
        for Xi, yi in self.client_data:
            err = Xi @ w - yi
            total_loss += float(np.sum(err**2))
        return total_loss / (2.0 * self.total_samples)

    def run_centralized(self, rounds: int = 50, lr: float = 0.02) -> Dict[str, Any]:
        """Centralized baseline: all data aggregated in single pool, gradient descent."""
        X_all = np.vstack([Xi for Xi, _ in self.client_data])
        y_all = np.concatenate([yi for _, yi in self.client_data])
        N = len(y_all)
        w = np.zeros(self.d, dtype=np.float64)
        losses = [self.evaluate_global_loss(w)]

        for r in range(rounds):
            grad = (X_all.T @ (X_all @ w - y_all)) / float(N)
            w -= lr * grad
            losses.append(self.evaluate_global_loss(w))

        return {
            "w_final": w.tolist(),
            "losses": losses,
            "communication_bytes": 0 # centralized baseline has 0 distributed network rounds
        }

    def run_fedavg(self, rounds: int = 50, local_steps: int = 5,
                   lr: float = 0.02) -> Dict[str, Any]:
        """
        FedAvg:
        1. Server broadcasts w_t to all clients.
        2. Clients perform local_steps of SGD.
        3. Server computes weighted average: w_{t+1} = sum (n_k / N) * w_k.
        """
        w_global = np.zeros(self.d, dtype=np.float64)
        losses = [self.evaluate_global_loss(w_global)]
        bytes_transferred = 0
        float_size = 8 # float64 = 8 bytes

        for r in range(rounds):
            client_weights = []
            for Xi, yi in self.client_data:
                # Downlink communication: client receives w_global
                bytes_transferred += self.d * float_size
                w_local = w_global.copy()
                n_i = len(yi)

                # Local optimization steps
                for step in range(local_steps):
                    grad = (Xi.T @ (Xi @ w_local - yi)) / float(n_i)
                    w_local -= lr * grad

                # Uplink communication: client uploads w_local
                bytes_transferred += self.d * float_size
                client_weights.append((n_i, w_local))

            # Server weighted aggregation
            w_next = np.zeros(self.d, dtype=np.float64)
            for n_i, w_k in client_weights:
                w_next += (n_i / self.total_samples) * w_k
            w_global = w_next
            losses.append(self.evaluate_global_loss(w_global))

        return {
            "local_steps": local_steps,
            "w_final": w_global.tolist(),
            "losses": losses,
            "communication_bytes": bytes_transferred
        }


class MultiTaskOptimizer:
    """
    Multi-Task Learning comparing:
    1. Equal Task Weighting: L_tot = 0.5 * L1 + 0.5 * L2
    2. MGDA (Multiple Gradient Descent Algorithm - Sener & Koltun, NeurIPS 2018)
    3. GradNorm (Chen et al., ICML 2018)
    """
    @staticmethod
    def solve_mgda_2task(g1: np.ndarray, g2: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Finds Pareto stationary direction min_{alpha in [0, 1]} ||alpha g1 + (1-alpha) g2||^2.
        For two tasks, the quadratic minimum has closed form solution:
        alpha = ((g2 - g1)^T g2) / ||g1 - g2||^2 clipped to [0, 1].
        """
        diff = g1 - g2
        diff_norm_sq = float(diff @ diff)
        if diff_norm_sq < 1e-12:
            return 0.5, 0.5, g1.copy()

        alpha = float((g2 - g1) @ g2) / diff_norm_sq
        alpha = np.clip(alpha, 0.0, 1.0)
        g_pareto = alpha * g1 + (1.0 - alpha) * g2
        return alpha, 1.0 - alpha, g_pareto

    @staticmethod
    def simulate_two_task_learning(n_steps: int = 150, lr: float = 0.05, seed: int = 42) -> Dict[str, Any]:
        """
        Two-task toy optimization problem with conflicting gradients:
        Task 1 objective: L1(theta) = 0.5 * (theta_1 - 2)^2 + (theta_2 - 2)^2
        Task 2 objective: L2(theta) = 0.5 * (theta_1 + 2)^2 + 4 * (theta_2 + 2)^2
        """
        def get_losses_and_grads(theta):
            L1 = 0.5 * (theta[0] - 2.0)**2 + (theta[1] - 2.0)**2
            g1 = np.array([theta[0] - 2.0, 2.0 * (theta[1] - 2.0)])

            L2 = 0.5 * (theta[0] + 2.0)**2 + 4.0 * (theta[1] + 2.0)**2
            g2 = np.array([theta[0] + 2.0, 8.0 * (theta[1] + 2.0)])
            return L1, g1, L2, g2

        methods = ["Equal_Weighting", "MGDA", "GradNorm"]
        results = {}

        for method in methods:
            theta = np.array([0.0, 0.0], dtype=np.float64)
            history_L1 = []
            history_L2 = []
            L1_0, _, L2_0, _ = get_losses_and_grads(theta)
            weights = np.array([0.5, 0.5])

            for step in range(n_steps):
                L1, g1, L2, g2 = get_losses_and_grads(theta)
                history_L1.append(float(L1))
                history_L2.append(float(L2))

                if method == "Equal_Weighting":
                    g_step = 0.5 * g1 + 0.5 * g2
                elif method == "MGDA":
                    _, _, g_step = MultiTaskOptimizer.solve_mgda_2task(g1, g2)
                elif method == "GradNorm":
                    # Balance gradient norms based on relative training loss ratios
                    norm1 = np.linalg.norm(g1) + 1e-8
                    norm2 = np.linalg.norm(g2) + 1e-8
                    avg_norm = (norm1 + norm2) / 2.0
                    r1 = (L1 / (L1_0 + 1e-8))
                    r2 = (L2 / (L2_0 + 1e-8))
                    avg_r = (r1 + r2) / 2.0
                    target1 = avg_norm * (r1 / avg_r)
                    target2 = avg_norm * (r2 / avg_r)
                    # Adaptive weights proportional to targets
                    w1 = target1 / (target1 + target2)
                    w2 = target2 / (target1 + target2)
                    g_step = w1 * g1 + w2 * g2

                theta -= lr * g_step

            results[method] = {
                "final_theta": theta.tolist(),
                "final_L1": history_L1[-1],
                "final_L2": history_L2[-1],
                "history_L1": history_L1,
                "history_L2": history_L2,
                "pareto_loss_sum": history_L1[-1] + history_L2[-1]
            }

        return results
