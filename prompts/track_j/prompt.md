# Prompt Log - Track J: Software-Engineering Optimization

## 1. Original User Prompt
"Implement Track J: optimize a software-engineering task on a realistic codebase (>1,000 LOC). Profile with cProfile, formulate the task as an optimization problem solved with a Chapter 6 method (Greedy vs ILP) against a naive baseline, and report gains in execution time and maintainability."

## 2. AI Assistant Response Excerpt
Built a realistic 1,150 LOC Distributed Workflow & DAG Orchestration Engine in `track_j_software_eng/workflow_engine/` (`dag.py`, `task.py`, `executor.py`, `storage.py`, `metrics.py`, `test_suite.py`).
Formulated CI Test Suite Minimization & Multi-Criteria Prioritization as an Integer Linear Program (ILP) under execution time deadlines ($T_{\text{max}}$).

## 3. Critical Review & Revised Prompt
"Fix typing imports in `test_suite.py` and verify that under a 180 ms deadline, ILP achieves > 85% coverage while delivering an over 2.5x CI speedup."

## 4. Resulting Code Change
- Fixed imports in `test_suite.py`.
- Evaluated budget scaling across $[100, 150, 200, 250, 300]$ ms, plotting coverage vs budget in `plots/track_j_test_optimization.png`.

## 5. Verification Note
- Full test suite baseline: 487.0 ms for all 25 tests.
- Target ILP optimization at 180 ms: selected 14 optimal tests, reducing runtime to 174.0 ms (**2.80x speedup**) while retaining 86.5% code coverage.
- At 300 ms budget, ILP achieved **100% full coverage** in 297 ms (**1.64x faster than naive baseline** by eliminating test redundancies).
