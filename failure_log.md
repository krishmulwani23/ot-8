# Mandatory AI Failure Log: 3 Documented Failure Cases & Lessons Learned

As required by **Section 8 (Mandatory Vibe-Coding Practice & AI-Tool Disclosure)** of Assignment 8, this log documents three real, substantive instances where AI-generated optimization code was wrong, unsafe, inefficient, or misleading. Each case includes the symptom, root cause analysis, the concrete fix, and the fundamental engineering lesson.

---

## Failure Case 1: Unsound Dead Code Elimination Stripping Side Effects (Track F: Compiler)

- **Track Affected**: Track F (Compiler Optimization passes)
- **Initial AI Prompt**: *"Write a Dead Code Elimination pass for Three-Address Code that identifies instructions whose destination variable is not subsequently used."*
- **Symptom / Manifestation**:
  When running the optimized TAC bytecode inside the virtual machine, program outputs vanished completely, and memory mutations were skipped. Programs failed semantic equivalence tests (`res_unopt["output"] != res_opt["output"]`).
- **Root Cause Analysis**:
  The AI assistant generated a naive backward sweep:
  ```python
  # AI GENERATED BUGGY CODE
  surviving = []
  for inst in instructions:
      if inst.dest in all_uses:
          surviving.append(inst)
  ```
  This implementation assumed that *any* instruction whose `dest` is not read in subsequent arithmetic expressions is dead. However, instructions such as `print`, `store`, `jmp_false`, and `ret` have critical external side effects (I/O, memory writes, control flow). Stripping them broke semantic equivalence.
- **Verification & Concrete Fix**:
  We introduced an explicit side-effect whitelist and an iterative fixed-point liveness solver:
  ```python
  # CORRECTED CODE
  for inst in current_insts:
      if inst.op in ('label', 'jmp', 'jmp_false', 'print', 'ret', 'store'):
          surviving.append(inst) # Side effects must NEVER be eliminated
      elif inst.dest and inst.dest not in all_uses:
          changed = True # Safely eliminate dead computation
      else:
          surviving.append(inst)
  ```
- **Engineering Lesson**:
  In compiler optimization passes, optimization without rigorous semantic verification is catastrophic. A compiler must preserve the *observable behavior* of the program under all circumstances. Automated equivalence assertions between unoptimized and optimized AST/IR execution must be run on every pass.

---

## Failure Case 2: Process State Machine Stalling and Burst Index Misalignment (Track G: xv6 OS)

- **Track Affected**: Track G (xv6 Operating System Scheduling)
- **Initial AI Prompt**: *"Implement the process state transitions in xv6 RISC-V for MLFQ where a process transitions to SLEEPING on I/O burst and ZOMBIE upon completion."*
- **Symptom / Manifestation**:
  During multi-process benchmark execution, the simulator hung in an infinite loop. CPU-bound processes were stuck in the process table, and `is_finished()` continually returned `False`, causing test timeout.
- **Root Cause Analysis**:
  In the AI-generated process lifecycle update:
  ```python
  # AI GENERATED BUGGY CODE
  if active_proc.current_cpu_rem <= 0:
      if active_proc.burst_idx < len(active_proc.io_bursts):
          active_proc.state = "SLEEPING"
          active_proc.current_io_rem = active_proc.io_bursts[active_proc.burst_idx]
      else:
          active_proc.state = "ZOMBIE"
          active_proc.completion_tick = self.ticks
          # BUG: Failed to increment burst_idx!
  ```
  The process `burst_idx` remained stuck at `len(cpu_bursts) - 1`. Because `is_finished()` checked `self.burst_idx >= len(self.cpu_bursts)`, the process was never considered finished by the scheduling harness, keeping the scheduler in an endless polling loop.
- **Verification & Concrete Fix**:
  Atomic progression of burst indices upon transitioning to `ZOMBIE`:
  ```python
  # CORRECTED CODE
  else:
      active_proc.state = "ZOMBIE"
      active_proc.completion_tick = self.ticks
      active_proc.burst_idx += 1 # Advance past final burst to mark completion
      active_proc = None
  ```
- **Engineering Lesson**:
  Operating system kernels rely on strict invariant state machines. When AI generates kernel or scheduler code, it frequently produces code that compiles cleanly but subtly corrupts internal accounting structures (e.g. tick counters, priority level pointers, or runqueue state). Every kernel change must be validated against automated state-transition unit tests.

---

## Failure Case 3: QAOA Simultaneous Multi-Gate Parameter Shift & Endianness Inversion (Track C: Quantum)

- **Track Affected**: Track C (Quantum Optimization: QAOA on Max-Cut QUBO)
- **Initial AI Prompt**: *"Implement QAOA for Max-Cut and calculate parameter gradients using the Parameter-Shift Rule."*
- **Symptom / Manifestation**:
  The QAOA optimization loop converged to the trivial bitstring `111111` with Cut = 0.0, completely failing to find the graph cut partition, despite classical solvers easily finding the maximum cut (8.0).
- **Root Cause Analysis**:
  Two deep compounding errors occurred in the AI code:
  1. **Qubit Endianness Mismatch**:
     In Python string formatting, `format(idx, "06b")` places the most significant bit ($2^{n-1}$, qubit 5) at string index 0, whereas the vertex mapping assumed string index $i$ corresponds to vertex $i$ (qubit 0). This flipped vertex assignments across the graph cut.
  2. **Simultaneous Multi-Gate Parameter-Shift Violation**:
     The standard Parameter-Shift Rule $\frac{f(\theta + \pi/2) - f(\theta - \pi/2)}{2}$ strictly applies to an isolated gate with generator eigenvalues $\pm 1/2$. In QAOA, parameter $\gamma$ is *shared* across all $|E|$ edge unitaries $e^{-i \gamma w_{uv} Z_u Z_v}$. Shifting $\gamma$ across all gates simultaneously does *not* yield the partial derivative—it introduces multi-frequency trigonometric cross-terms that completely scrambled the gradient descent trajectory.
- **Verification & Concrete Fix**:
  1. We corrected bitstring decoding using explicit bit-shift extraction:
     ```python
     # Corrected endianness: qubit q corresponds to (idx >> q) & 1
     bit_arr = [(best_int >> q) & 1 for q in range(self.n_qubits)]
     ```
  2. We switched parameter optimization to Simultaneous Perturbation Stochastic Approximation (SPSA) and exact finite-difference gradient evaluation, which naturally handle multi-gate parameter sharing without requiring individual gate-level chain rule unrolling.
- **Engineering Lesson**:
  Quantum computing algorithms have subtle mathematical physics prerequisites that LLMs frequently gloss over. One cannot treat parameterized quantum circuits as generic differentiable functions without respecting the operator algebra and generator spectra of quantum gates.
