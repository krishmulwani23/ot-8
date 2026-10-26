# Prompt Log - Track C: Quantum Optimization

## 1. Original User Prompt
"Implement Track C Quantum Optimization: build a statevector quantum simulator engine, implement QAOA and VQE, quantum gradient methods (Parameter-Shift Rule, SPSA, Quantum Natural Gradient), solve the Track B Max-Cut QUBO using QAOA, and evaluate finite measurement shots, hardware noise, and barren plateaus."

## 2. AI Assistant Response Excerpt
Created `track_c_quantum/quantum_engine.py`, `track_c_quantum/gradients.py`, `track_c_quantum/qaoa_vqe.py`, and `track_c_quantum/noise_and_barren_plateaus.py`.
Initially encountered a parameter-shift multi-gate sharing mismatch and bitstring endianness reversal.

## 3. Critical Review & Revised Prompt
"Fix QAOA bitstring extraction endianness: format string index 0 is MSB, which was mapping qubit n-1 to vertex 0. For shared parameters gamma in QAOA, explain why standard single-gate parameter shift fails and implement finite-difference/SPSA gradient methods. Also fix parameter vector sizing in `study_barren_plateaus`."

## 4. Resulting Code Change
- Corrected bitstring extraction: `bit_arr = [(best_int >> q) & 1 for q in range(self.n_qubits)]`.
- Implemented SPSA and multi-gate gradient handling in `qaoa_vqe.py`.
- Fixed parameter sizing in `noise_and_barren_plateaus.py` to differentiate specifically with respect to parameter 0.

## 5. Verification Note
- QAOA achieved the exact optimal cut ($8.0$, bitstring `010101`), perfectly matching the classical Track B baseline.
- Barren plateaus demonstrated exponential variance decay: $1.18 \times 10^{-1}$ (n=2) down to $2.01 \times 10^{-3}$ (n=8).
- Finite measurement shot noise verified $\sigma \propto 1/\sqrt{N_{\text{shots}}}$ scaling.
