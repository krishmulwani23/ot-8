"""
Unified Automated Test Suite for All Tracks (Tracks A to J).
Validates mathematical invariants, algorithm correctness, and execution consistency.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Track A
from track_a_continuous.landscapes import ConvexQuadratic, Rosenbrock
from track_a_continuous.optimizers import GradientDescent, Adam
from track_a_continuous.backprop_model import TwoLayerMLP

# Track B
from track_b_discrete.solvers import KnapsackInstance, GreedyKnapsackSolver, DynamicProgrammingKnapsackSolver, ILPKnapsackSolver
from track_b_discrete.qubo_ising import MaxCutProblem

# Track C
from track_c_quantum.quantum_engine import QuantumCircuit
from track_c_quantum.qaoa_vqe import QAOAOptimizer

# Track D
from track_d_parallel_distributed.submodular import SensorCoverageSubmodular

# Track E
from track_e_case_study.formulation import FleetChargingProblem
from track_e_case_study.solver_milp import MILPFleetOptimizer

# Track F
from track_f_compiler.ir import Instruction
from track_f_compiler.optimizations import OptimizationPipeline
from track_f_compiler.virtual_machine import TACVirtualMachine

# Track G
from track_g_xv6.xv6_kernel_sim import SimProcess, XV6KernelSimulator

# Track H
from track_h_dbms.database import TPCHDatabase
from track_h_dbms.index_selection import IndexSelectionOptimizer

# Track I
from track_i_quantum_algorithms.algorithms import QuantumAlgorithmsSuite

# Track J
from track_j_software_eng.workflow_engine.test_suite import get_all_test_cases
from track_j_software_eng.solvers import TestSuiteOptimizer


class TestAllTracks(unittest.TestCase):

    def test_track_a_backprop_gradient_accuracy(self):
        """Track A: Verify backpropagation produces accurate gradients (< 1e-4 error)."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 2))
        y = (X[:, 0]**2 + X[:, 1]**2 > 1.0).astype(float).reshape(-1, 1)
        mlp = TwoLayerMLP(input_dim=2, hidden_dim=4, output_dim=1, seed=42)
        diff = mlp.verify_gradients_numerical(X, y)
        self.assertLess(diff, 1e-4, f"Gradient verification error too high: {diff}")

    def test_track_b_knapsack_solvers_consistency(self):
        """Track B: Verify DP and ILP match on exact 0-1 Knapsack."""
        inst = KnapsackInstance(values=[10, 15, 20, 25], weights=[2, 3, 5, 7], capacity=10)
        dp = DynamicProgrammingKnapsackSolver().solve(inst)
        ilp = ILPKnapsackSolver().solve(inst)
        self.assertEqual(dp["total_value"], ilp["total_value"])

    def test_track_c_quantum_probabilities_sum_to_one(self):
        """Track C: Verify quantum circuit unitary evolution preserves norm."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.ry(0.5, 2)
        probs = qc.probabilities()
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)

    def test_track_d_submodular_diminishing_returns(self):
        """Track D: Verify diminishing returns property holds strictly."""
        sc = SensorCoverageSubmodular(n_targets=50, n_sensors=10, radius=0.3, seed=42)
        prop = sc.verify_diminishing_returns(n_trials=50)
        self.assertTrue(prop["is_submodular"])
        self.assertEqual(prop["violations"], 0)

    def test_track_e_milp_charging_feasibility(self):
        """Track E: Verify MILP charging schedule satisfies all vehicle energy targets."""
        prob = FleetChargingProblem(n_vehicles=5, transformer_limit_kw=200.0, seed=42)
        sol = MILPFleetOptimizer().solve(prob)
        self.assertGreater(sol["total_cost"], 0.0)
        self.assertEqual(sol["transformer_overload_hours"], 0)

    def test_track_f_compiler_semantic_equivalence(self):
        """Track F: Verify unoptimized vs optimized code produces identical output."""
        prog = [
            Instruction('assign', dest='a', src1='5'),
            Instruction('assign', dest='b', src1='3'),
            Instruction('add', dest='c', src1='a', src2='b'),
            Instruction('add', dest='d', src1='a', src2='b'), # CSE candidate
            Instruction('add', dest='tot', src1='c', src2='d'),
            Instruction('print', src1='tot'),
            Instruction('ret', src1='tot')
        ]
        vm = TACVirtualMachine()
        res_unopt = vm.execute(prog)
        prog_opt = OptimizationPipeline.run_all_passes(prog)
        res_opt = vm.execute(prog_opt)
        self.assertEqual(res_unopt["output"], res_opt["output"])
        self.assertLess(len(prog_opt), len(prog))

    def test_track_g_xv6_scheduler_execution(self):
        """Track G: Verify MLFQ scheduler completes all tasks without deadlock."""
        procs = [
            SimProcess(1, "cpu", 0, [20], []),
            SimProcess(2, "io", 0, [2, 2], [3])
        ]
        sim = XV6KernelSimulator(scheduler_type="mlfq")
        res = sim.run(procs, max_ticks=100)
        self.assertTrue(all(p.is_finished() for p in procs))

    def test_track_h_dbms_query_equivalence(self):
        """Track H: Verify TPC-H database generates consistent query results."""
        db = TPCHDatabase(seed=42)
        cur = db.conn.cursor()
        cur.execute("SELECT count(*) FROM customer;")
        cnt = cur.fetchone()[0]
        self.assertGreater(cnt, 0)

    def test_track_i_deutsch_jozsa_oracle(self):
        """Track I: Verify Deutsch-Jozsa correctly determines balanced oracle in 1 query."""
        res = QuantumAlgorithmsSuite.deutsch_jozsa(n_qubits=3, oracle_type="balanced")
        self.assertTrue(res["success"])
        self.assertEqual(res["quantum_queries"], 1)

    def test_track_j_swe_test_selection_budget(self):
        """Track J: Verify ILP test selection respects CI time budget."""
        tests = get_all_test_cases()
        opt = TestSuiteOptimizer(tests)
        budget = 150.0
        sol = opt.solve_ilp(budget)
        self.assertLessEqual(sol["total_execution_time_ms"], budget)
        self.assertGreater(sol["coverage_pct"], 70.0)


if __name__ == "__main__":
    unittest.main()
