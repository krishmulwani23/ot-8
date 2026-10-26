# Prompt Log - Track I: Quantum Algorithms and QML Advantage

## 1. Original User Prompt
"Implement Track I: run every quantum algorithm taught in Chapter 6 (Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor, Quantum Annealing). Train a Variational Quantum Classifier (VQC) and an equivalent classical MLP on the same dataset, compare accuracy, parameters, training time, and circuit evaluations. Analyze algorithmic advantage (Grover vs classical linear search) and discuss simulation overhead, noise, and barren plateaus."

## 2. AI Assistant Response Excerpt
Created `track_i_quantum_algorithms/algorithms.py`, `track_i_quantum_algorithms/qml_models.py`, and `track_i_quantum_algorithms/experiments.py`.
Implemented all 7 quantum algorithms from scratch without external black-box library wrappers. Evaluated VQC and classical MLP on non-linear circular classification.

## 3. Critical Review & Revised Prompt
"Ensure the classical simulation bottleneck is explained honestly as a negative result, noting why simulating 2^n statevectors on classical CPUs is asymptotically slower than parallel classical algorithms."

## 4. Resulting Code Change
- Added explicit negative results analysis documenting $O(2^n)$ classical simulation overhead, QRAM data loading bottlenecks, and barren plateaus.
- Plotted Grover quadratic query advantage ($\frac{\pi}{4}\sqrt{N}$ vs $N/2$) in `plots/track_i_grover_scaling.png`.

## 5. Verification Note
- All 7 quantum algorithms passed execution verification.
- VQC achieved 73.8% accuracy with only 6 parameters, matching the classical MLP (73.8% with 17 parameters), but requiring 2,576 circuit evaluations (459.5 ms vs 7.3 ms for classical backprop).
