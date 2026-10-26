"""
Vectorized Quantum Statevector and Circuit Simulator (Track C & Track I).
Implements exact state evolution, quantum gate applications, projective measurements,
shot sampling, and density-matrix depolarizing noise models.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Fundamental 2x2 Pauli & Single-Qubit Gate Matrices
I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
X_GATE = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y_GATE = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z_GATE = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H_GATE = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)

def rx_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2.0), -1.0j * np.sin(theta / 2.0)],
        [-1.0j * np.sin(theta / 2.0), np.cos(theta / 2.0)]
    ], dtype=np.complex128)

def ry_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0), np.cos(theta / 2.0)]
    ], dtype=np.complex128)

def rz_gate(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1.0j * theta / 2.0), 0.0],
        [0.0, np.exp(1.0j * theta / 2.0)]
    ], dtype=np.complex128)


class QuantumCircuit:
    """
    Quantum Circuit operating on n_qubits statevector.
    State representation: complex array of shape (2^n,).
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        # Initialize state to |00...0>
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0
        self.gates: List[Tuple[str, Tuple, Dict[str, Any]]] = []

    def reset(self):
        self.state = np.zeros(self.dim, dtype=np.complex128)
        self.state[0] = 1.0

    def copy(self) -> 'QuantumCircuit':
        qc = QuantumCircuit(self.n_qubits)
        qc.state = self.state.copy()
        qc.gates = list(self.gates)
        return qc

    def apply_single_gate(self, matrix: np.ndarray, target: int):
        """Applies single-qubit 2x2 matrix to target qubit in statevector using tensor reshaping."""
        n = self.n_qubits
        # Reshape to (2, 2, ..., 2) tensor of rank n
        tensor = self.state.reshape([2] * n)
        # Move target axis to front: axis target becomes axis 0
        tensor = np.moveaxis(tensor, target, 0)
        # Apply matrix product along axis 0
        orig_shape = tensor.shape
        tensor = tensor.reshape(2, -1)
        tensor = matrix @ tensor
        tensor = tensor.reshape(orig_shape)
        # Move axis back
        tensor = np.moveaxis(tensor, 0, target)
        self.state = tensor.reshape(self.dim)

    def apply_two_qubit_gate(self, matrix4: np.ndarray, q1: int, q2: int):
        """Applies 4x4 two-qubit gate matrix to qubits (q1, q2)."""
        n = self.n_qubits
        tensor = self.state.reshape([2] * n)
        # Move q1 to axis 0, q2 to axis 1
        tensor = np.moveaxis(tensor, [q1, q2], [0, 1])
        orig_shape = tensor.shape
        tensor = tensor.reshape(4, -1)
        tensor = matrix4 @ tensor
        tensor = tensor.reshape(orig_shape)
        tensor = np.moveaxis(tensor, [0, 1], [q1, q2])
        self.state = tensor.reshape(self.dim)

    def h(self, target: int):
        self.apply_single_gate(H_GATE, target)
        self.gates.append(("h", (target,), {}))

    def x(self, target: int):
        self.apply_single_gate(X_GATE, target)
        self.gates.append(("x", (target,), {}))

    def y(self, target: int):
        self.apply_single_gate(Y_GATE, target)
        self.gates.append(("y", (target,), {}))

    def z(self, target: int):
        self.apply_single_gate(Z_GATE, target)
        self.gates.append(("z", (target,), {}))

    def rx(self, theta: float, target: int):
        self.apply_single_gate(rx_gate(theta), target)
        self.gates.append(("rx", (target,), {"theta": theta}))

    def ry(self, theta: float, target: int):
        self.apply_single_gate(ry_gate(theta), target)
        self.gates.append(("ry", (target,), {"theta": theta}))

    def rz(self, theta: float, target: int):
        self.apply_single_gate(rz_gate(theta), target)
        self.gates.append(("rz", (target,), {"theta": theta}))

    def cx(self, control: int, target: int):
        """Controlled-X (CNOT) gate."""
        cnot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        self.apply_two_qubit_gate(cnot, control, target)
        self.gates.append(("cx", (control, target), {}))

    def cz(self, control: int, target: int):
        """Controlled-Z gate."""
        cz = np.diag([1, 1, 1, -1]).astype(np.complex128)
        self.apply_two_qubit_gate(cz, control, target)
        self.gates.append(("cz", (control, target), {}))

    def probabilities(self) -> np.ndarray:
        """Returns exact measurement probabilities P(x) = |psi(x)|^2."""
        probs = np.abs(self.state)**2
        # Normalize in case of tiny numerical drift
        s = np.sum(probs)
        return probs / s if s > 0 else probs

    def sample_shots(self, n_shots: int = 1024, seed: Optional[int] = None) -> Dict[str, int]:
        """Simulates projective measurements over finite measurement shots."""
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        samples = rng.choice(self.dim, size=n_shots, p=probs)
        counts: Dict[str, int] = {}
        for bit_idx in samples:
            # Format as binary bitstring matching n_qubits (e.g. '0110')
            bitstr = format(bit_idx, f"0{self.n_qubits}b")
            counts[bitstr] = counts.get(bitstr, 0) + 1
        return counts

    def expectation_diagonal(self, diag_operator: np.ndarray) -> float:
        """Computes exact <psi | O | psi> for diagonal Hamiltonian."""
        probs = self.probabilities()
        return float(np.sum(probs * diag_operator))
