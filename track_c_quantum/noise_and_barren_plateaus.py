"""
Noise Models, Finite-Shot Measurement Statistics, and Barren Plateaus (Track C).
Investigates:
1. Finite-shot sampling variance scaling ~ 1 / sqrt(N_shots)
2. Hardware depolarizing noise and fidelity decay
3. Barren Plateaus: exponential decay of gradient variance Var[grad] ~ O(2^{-n})
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from track_c_quantum.quantum_engine import QuantumCircuit
from track_c_quantum.gradients import QuantumGradients

class QuantumNoiseAndScaling:
    @staticmethod
    def finite_shot_analysis(qc: QuantumCircuit, diag_observable: np.ndarray,
                             shot_counts: List[int] = [100, 500, 2000, 10000],
                             trials: int = 30) -> Dict[str, Any]:
        """
        Studies standard deviation of expectation value as a function of measurement shots.
        Verifies 1 / sqrt(N_shots) quantum projection noise law.
        """
        exact_exp = qc.expectation_diagonal(diag_observable)
        results = {}

        for n_shots in shot_counts:
            estimates = []
            for t in range(trials):
                counts = qc.sample_shots(n_shots=n_shots, seed=1000 * n_shots + t)
                # Compute empirical expectation from shot counts
                emp_exp = 0.0
                for bitstr, cnt in counts.items():
                    idx = int(bitstr, 2)
                    emp_exp += (cnt / n_shots) * diag_observable[idx]
                estimates.append(emp_exp)

            std_err = float(np.std(estimates))
            mean_est = float(np.mean(estimates))
            results[str(n_shots)] = {
                "mean_estimate": mean_est,
                "exact_expectation": float(exact_exp),
                "std_deviation": std_err,
                "theoretical_shot_noise": 1.0 / np.sqrt(n_shots)
            }

        return results

    @staticmethod
    def simulate_depolarizing_channel(state: np.ndarray, p_noise: float) -> np.ndarray:
        """
        Simulates depolarizing noise channel on pure state:
        rho' = (1 - p) |psi><psi| + p * I / (2^n)
        Returns diagonal elements of density matrix (probabilities).
        """
        dim = len(state)
        probs_pure = np.abs(state)**2
        probs_noisy = (1.0 - p_noise) * probs_pure + p_noise * (1.0 / dim)
        return probs_noisy

    @staticmethod
    def study_barren_plateaus(qubit_counts: List[int] = [2, 4, 6, 8],
                              depth: int = 3, n_samples: int = 40) -> Dict[str, Any]:
        """
        Demonstrates the Barren Plateau phenomenon (McClean et al., Nature Communications 2018):
        For random parameterized quantum circuits, the variance of the gradient
        decays exponentially with the number of qubits: Var[d<H>/d theta] ~ O(2^{-n}).
        """
        rng = np.random.default_rng(42)
        variances = []
        mean_abs_grads = []

        for n in qubit_counts:
            # Measure local Pauli Z on qubit 0
            dim = 1 << n
            diag_z0 = np.array([1.0 if ((idx & 1) == 0) else -1.0 for idx in range(dim)])
            grads = []

            for s in range(n_samples):
                # Build random hardware-efficient circuit with 2*depth*n parameters
                n_params = 2 * depth * n
                params = rng.uniform(0, 2*np.pi, size=n_params)

                def circuit_eval(p: np.ndarray) -> float:
                    qc = QuantumCircuit(n)
                    p_idx = 0
                    for d in range(depth):
                        # Rotation layer
                        for q in range(n):
                            qc.ry(p[p_idx], q)
                            qc.rz(p[p_idx + 1], q)
                            p_idx += 2
                        # Entangling ring layer
                        for q in range(n):
                            qc.cx(q, (q + 1) % n)
                    return qc.expectation_diagonal(diag_z0)

                # Compute gradient with respect to the parameters using parameter-shift rule
                # Differentiate specifically with respect to the first parameter (theta_0)
                p_pos = params.copy()
                p_neg = params.copy()
                shift = np.pi / 2.0
                p_pos[0] += shift
                p_neg[0] -= shift
                exp_pos = circuit_eval(p_pos)
                exp_neg = circuit_eval(p_neg)
                g0 = (exp_pos - exp_neg) / (2.0 * np.sin(shift))
                grads.append(float(g0))

            var_g = float(np.var(grads))
            mean_g = float(np.mean(np.abs(grads)))
            variances.append(var_g)
            mean_abs_grads.append(mean_g)
            print(f"  Qubits n={n}: Var[grad] = {var_g:.6e}, Mean(|grad|) = {mean_g:.6e}")

        return {
            "qubit_counts": qubit_counts,
            "gradient_variances": variances,
            "mean_abs_gradients": mean_abs_grads
        }
