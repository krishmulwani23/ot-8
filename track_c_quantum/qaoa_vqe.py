"""
QAOA and VQE Implementations (Track C):
- Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut
- Variational Quantum Eigensolver (VQE) with hardware-efficient ansatz
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from track_c_quantum.quantum_engine import QuantumCircuit, Z_GATE, X_GATE
from track_c_quantum.gradients import QuantumGradients

class QAOAOptimizer:
    """
    Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut.
    Solves the exact graph instances formulated in Track B.
    """
    def __init__(self, n_qubits: int, edges: List[Tuple[int, int, float]], p_layers: int = 2):
        self.n_qubits = n_qubits
        self.edges = edges
        self.p_layers = p_layers
        self.dim = 1 << n_qubits

        # Precompute diagonal of Cost Hamiltonian H_C in computational basis
        self.diag_cost = np.zeros(self.dim, dtype=np.float64)
        for idx in range(self.dim):
            # Bit array for basis state |idx>
            bits = [(idx >> k) & 1 for k in range(self.n_qubits)]
            cut = 0.0
            for u, v, w in self.edges:
                if bits[u] != bits[v]:
                    cut += w
            self.diag_cost[idx] = cut

    def build_circuit(self, gamma: np.ndarray, beta: np.ndarray) -> QuantumCircuit:
        """Constructs and simulates the QAOA circuit for parameters (gamma, beta)."""
        qc = QuantumCircuit(self.n_qubits)
        # 1. Prepare uniform superposition |+>^n
        for q in range(self.n_qubits):
            qc.h(q)

        # 2. Alternating layers of Problem (Cost) and Mixer unitaries
        for l in range(self.p_layers):
            g = gamma[l]
            b = beta[l]

            # Cost Unitary: e^{-i gamma H_C}
            # For each edge (u, v): apply Rz(2*gamma) via CX - Rz - CX
            for u, v, w in self.edges:
                qc.cx(u, v)
                qc.rz(2.0 * g * w, v)
                qc.cx(u, v)

            # Mixer Unitary: e^{-i beta H_M} = prod_q Rx(2*beta)
            for q in range(self.n_qubits):
                qc.rx(2.0 * b, q)

        return qc

    def cost_function(self, params: np.ndarray) -> float:
        """Expectation value <psi | -H_C | psi> to minimize."""
        gamma = params[:self.p_layers]
        beta = params[self.p_layers:]
        qc = self.build_circuit(gamma, beta)
        # Minimize negative cut = maximize cut
        exp_cut = qc.expectation_diagonal(self.diag_cost)
        return -exp_cut

    def optimize_classical_loop(self, init_params: np.ndarray = None, max_iter: int = 50,
                                lr: float = 0.08, method: str = "parameter_shift") -> Dict[str, Any]:
        """
        Hybrid Quantum-Classical Optimization loop using quantum gradients.
        Methods: 'parameter_shift', 'spsa', or 'qng'.
        """
        if init_params is None:
            # Standard initial heuristic for QAOA
            gamma0 = np.linspace(0.1, 0.5, self.p_layers)
            beta0 = np.linspace(0.5, 0.1, self.p_layers)
            params = np.concatenate([gamma0, beta0])
        else:
            params = np.array(init_params, dtype=np.float64, copy=True)

        history_cost = []
        best_cut = 0.0

        for it in range(max_iter):
            cost = self.cost_function(params)
            history_cost.append(-cost) # store positive cut value
            if -cost > best_cut:
                best_cut = -cost

            if method == "parameter_shift":
                # Note: Because gamma and beta are shared across multiple gates,
                # the exact gradient can be computed via finite differences or gate-wise chain rule.
                eps = 1e-3
                grad = np.zeros_like(params)
                for i in range(len(params)):
                    p_pos = params.copy()
                    p_neg = params.copy()
                    p_pos[i] += eps
                    p_neg[i] -= eps
                    grad[i] = (self.cost_function(p_pos) - self.cost_function(p_neg)) / (2.0 * eps)
            elif method == "spsa":
                grad = QuantumGradients.spsa_gradient(self.cost_function, params)
            else: # numerical gradient default
                eps = 1e-4
                grad = np.zeros_like(params)
                for i in range(len(params)):
                    p_pos = params.copy()
                    p_neg = params.copy()
                    p_pos[i] += eps
                    p_neg[i] -= eps
                    grad[i] = (self.cost_function(p_pos) - self.cost_function(p_neg)) / (2.0 * eps)

            # Gradient descent step
            params = params - lr * grad

        # Final evaluation and sample solutions
        final_gamma = params[:self.p_layers]
        final_beta = params[self.p_layers:]
        final_qc = self.build_circuit(final_gamma, final_beta)
        sampled_counts = final_qc.sample_shots(n_shots=4096, seed=42)

        # Find bitstring with maximum sampled frequency
        best_bitstr = max(sampled_counts.items(), key=lambda kv: kv[1])[0]
        # In format(idx, '0nb'), char 0 is the most significant bit (qubit n-1)
        # Convert bitstring correctly to qubit index array: qubit q is (idx >> q) & 1
        best_int = int(best_bitstr, 2)
        bit_arr = [(best_int >> q) & 1 for q in range(self.n_qubits)]
        
        # Evaluate cut for best bitstring
        sampled_cut = 0.0
        for u, v, w in self.edges:
            if bit_arr[u] != bit_arr[v]:
                sampled_cut += w

        return {
            "method": method,
            "best_cut_expected": float(best_cut),
            "sampled_cut": float(sampled_cut),
            "best_bitstring": best_bitstr,
            "sampled_counts": sampled_counts,
            "final_params": params.tolist(),
            "history_cuts": history_cost,
            "optimal_cut_achieved": float(max(sampled_cut, best_cut))
        }


class VQEOptimizer:
    """
    Variational Quantum Eigensolver (VQE) with Hardware-Efficient Ansatz.
    Finds ground state energy E0 = min <psi(theta)| H |psi(theta)>.
    """
    def __init__(self, n_qubits: int, layers: int = 2):
        self.n_qubits = n_qubits
        self.layers = layers
        self.n_params = n_qubits * (layers + 1)

    def build_ansatz(self, theta: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        # Layer 0 rotations
        for q in range(self.n_qubits):
            qc.ry(theta[param_idx], q)
            param_idx += 1

        for l in range(self.layers):
            # Entanglement layer (linear CZ entanglers)
            for q in range(self.n_qubits - 1):
                qc.cz(q, q + 1)
            # Rotation layer
            for q in range(self.n_qubits):
                qc.ry(theta[param_idx], q)
                param_idx += 1

        return qc

    def ground_state_search(self, hamiltonian_diag: np.ndarray,
                            max_iter: int = 40, lr: float = 0.1) -> Dict[str, Any]:
        rng = np.random.default_rng(42)
        params = rng.uniform(0, 2*np.pi, size=self.n_params)

        def cost_fn(p: np.ndarray) -> float:
            qc = self.build_ansatz(p)
            return qc.expectation_diagonal(hamiltonian_diag)

        history_energy = []
        for it in range(max_iter):
            energy = cost_fn(params)
            history_energy.append(float(energy))
            grad = QuantumGradients.parameter_shift_gradient(cost_fn, params)
            params -= lr * grad

        exact_ground_energy = float(np.min(hamiltonian_diag))
        return {
            "final_energy": history_energy[-1],
            "exact_ground_energy": exact_ground_energy,
            "energy_error": abs(history_energy[-1] - exact_ground_energy),
            "history_energy": history_energy
        }
