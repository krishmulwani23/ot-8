"""
Experimental benchmark suite for Track A: Continuous Optimization.
Runs all comparisons, verifies backpropagation accuracy, logs results,
and outputs high-resolution convergence figures.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

# Ensure matplotlib works in headless sandbox
os.environ["MPLCONFIGDIR"] = os.path.abspath(".matplotlib_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from track_a_continuous.landscapes import (
    ConvexQuadratic, SaddlePointLandscape, Rosenbrock, Rastrigin, NoisyObjective
)
from track_a_continuous.optimizers import (
    GradientDescent, MomentumOptimizer, NesterovMomentum,
    AdaGrad, RMSProp, Adam, MiniBatchSGDOptimizer
)
from track_a_continuous.second_order import (
    NewtonOptimizer, GaussNewtonOptimizer, CurvatureApproximations
)
from track_a_continuous.backprop_model import TwoLayerMLP


def run_landscape_experiments():
    print("[Track A] Running Continuous Optimizer Comparisons across landscapes...")
    results = {}

    landscapes = [
        ("Convex_Quadratic", ConvexQuadratic(kappa=15.0), np.array([4.0, 4.0])),
        ("Rosenbrock", Rosenbrock(a=1.0, b=10.0), np.array([-1.2, 1.0])),
        ("Saddle_Point", SaddlePointLandscape(), np.array([0.5, 0.01])),
        ("Rastrigin", Rastrigin(dim=2), np.array([1.5, 2.5]))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, landscape, x0) in enumerate(landscapes):
        results[name] = {}
        ax = axes[idx]

        optimizers = [
            ("GD", GradientDescent(lr=0.01, max_iter=200)),
            ("Momentum", MomentumOptimizer(lr=0.01, beta=0.9, max_iter=200)),
            ("NAG", NesterovMomentum(lr=0.01, beta=0.9, max_iter=200)),
            ("AdaGrad", AdaGrad(lr=0.2, max_iter=200)),
            ("RMSProp", RMSProp(lr=0.02, beta=0.9, max_iter=200)),
            ("Adam", Adam(lr=0.05, beta1=0.9, beta2=0.999, max_iter=200)),
        ]

        # Add Newton for compatible functions
        if name != "Rastrigin": # Newton on highly oscillatory Rastrigin requires line search
            newton = NewtonOptimizer(lr=0.8, reg=1e-3, max_iter=100)
            t0 = time.perf_counter()
            n_res = newton.optimize(landscape.value, landscape.gradient, landscape.hessian, x0)
            t_el = (time.perf_counter() - t0) * 1000.0
            results[name]["Newton"] = {
                "final_loss": float(n_res["losses"][-1]),
                "iterations": n_res["iterations"],
                "runtime_ms": t_el,
                "x_final": n_res["x_final"].tolist()
            }
            ax.plot(n_res["losses"], label="Newton", linewidth=2.2, linestyle="--")

        for opt_name, opt in optimizers:
            t0 = time.perf_counter()
            state = opt.optimize(landscape.value, landscape.gradient, x0)
            t_el = (time.perf_counter() - t0) * 1000.0

            results[name][opt_name] = {
                "final_loss": float(state.losses[-1]),
                "iterations": state.iterations,
                "runtime_ms": t_el,
                "x_final": state.x.tolist()
            }
            ax.plot(state.losses, label=opt_name, linewidth=1.6)

        ax.set_title(f"Landscape: {name.replace('_', ' ')}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Loss / Objective", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_a_convergence.png", dpi=300)
    plt.close()
    print("  Saved plots/track_a_convergence.png")
    return results


def run_backpropagation_and_ml_experiment():
    print("[Track A] Running Backpropagation & Neural Network Optimization Experiment...")
    rng = np.random.default_rng(123)
    N = 200
    # Generate non-linear circular classification dataset
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2*np.pi, N)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    y = ((r > 0.5).astype(np.float64)).reshape(-1, 1)

    mlp = TwoLayerMLP(input_dim=2, hidden_dim=8, output_dim=1, reg_lambda=1e-3, seed=42)
    diff = mlp.verify_gradients_numerical(X, y)
    print(f"  Backprop relative gradient verification error: {diff:.3e} (PASS: < 1e-4)")
    assert diff < 1e-4, f"Backprop numerical gradient verification failed: diff={diff}"

    # Compare optimizers training the neural network using backpropagation gradients
    opt_configs = [
        ("GD", GradientDescent(lr=0.1, max_iter=250)),
        ("Momentum", MomentumOptimizer(lr=0.08, beta=0.9, max_iter=250)),
        ("Adam", Adam(lr=0.03, beta1=0.9, beta2=0.999, max_iter=250)),
        ("RMSProp", RMSProp(lr=0.02, max_iter=250))
    ]

    theta0 = mlp.get_flat_params()
    fig, ax = plt.subplots(figsize=(8, 5))
    ml_results = {"grad_verification_error": diff, "optimizers": {}}

    for opt_name, opt in opt_configs:
        def func(w):
            l, _ = mlp.compute_loss_and_flat_grad(w, X, y)
            return l

        def grad_fn(w):
            _, g = mlp.compute_loss_and_flat_grad(w, X, y)
            return g

        t0 = time.perf_counter()
        state = opt.optimize(func, grad_fn, theta0)
        t_ms = (time.perf_counter() - t0) * 1000.0

        ml_results["optimizers"][opt_name] = {
            "final_loss": float(state.losses[-1]),
            "runtime_ms": t_ms
        }
        ax.plot(state.losses, label=f"{opt_name} (final: {state.losses[-1]:.4f})", linewidth=1.8)

    ax.set_title("MLP Training on Non-linear Dataset (Backprop Gradients)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch / Iteration", fontsize=10)
    ax.set_ylabel("Binary Cross Entropy Loss", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/track_a_backprop_ml.png", dpi=300)
    plt.close()
    print("  Saved plots/track_a_backprop_ml.png")
    return ml_results


def run_curvature_and_hvp_experiment():
    print("[Track A] Running Curvature Approximations & Hessian-Vector Product (HVP) Test...")
    obj = ConvexQuadratic(kappa=20.0)
    x = np.array([2.5, -1.8])
    H_true = obj.hessian(x)

    # 1. Hessian-Vector Product check
    v = np.array([0.6, 0.8])
    hvp_approx = obj.hessian_vector_product(x, v)
    hvp_exact = H_true @ v
    hvp_err = float(np.linalg.norm(hvp_approx - hvp_exact))
    print(f"  HVP approximation relative error: {hvp_err:.3e}")

    # 2. Diagonal approximation error
    H_diag = CurvatureApproximations.diagonal_hessian(H_true)
    diag_frob_err = float(np.linalg.norm(H_true - H_diag, 'fro'))

    # 3. Truncated Newton-CG step using purely HVP
    grad = obj.gradient(x)
    step_cg = CurvatureApproximations.newton_cg_step(grad, lambda vec: obj.hessian_vector_product(x, vec))
    step_exact = np.linalg.solve(H_true, grad)
    cg_err = float(np.linalg.norm(step_cg - step_exact))
    print(f"  Newton-CG Hessian-free step error vs exact solve: {cg_err:.3e}")

    return {
        "hvp_error": hvp_err,
        "diagonal_approx_frobenius_norm": diag_frob_err,
        "newton_cg_step_error": cg_err
    }


def run_noisy_and_learning_rate_sensitivity():
    print("[Track A] Running Learning Rate Tuning & Noisy Objective Sensitivity...")
    base_obj = ConvexQuadratic(kappa=10.0)
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    x0 = np.array([3.0, 3.0])

    lr_results = {}
    for lr in lrs:
        adam = Adam(lr=lr, max_iter=150)
        st_clean = adam.optimize(base_obj.value, base_obj.gradient, x0)

        noisy_obj = NoisyObjective(base_obj, noise_std=0.2, seed=42)
        adam_noisy = Adam(lr=lr, max_iter=150)
        st_noisy = adam_noisy.optimize(noisy_obj.value, noisy_obj.gradient, x0)

        lr_results[str(lr)] = {
            "clean_final_loss": float(st_clean.losses[-1]),
            "noisy_final_loss": float(st_noisy.losses[-1])
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    clean_losses = [lr_results[str(lr)]["clean_final_loss"] for lr in lrs]
    noisy_losses = [lr_results[str(lr)]["noisy_final_loss"] for lr in lrs]
    ax.plot(lrs, clean_losses, marker="o", label="Clean Gradient", linewidth=2)
    ax.plot(lrs, noisy_losses, marker="s", label="Noisy Gradient (sigma=0.2)", linewidth=2, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Adam Learning Rate Sensitivity & Noise Robustness", fontsize=12, fontweight="bold")
    ax.set_xlabel("Learning Rate", fontsize=10)
    ax.set_ylabel("Final Objective Value", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/track_a_lr_sensitivity.png", dpi=300)
    plt.close()
    print("  Saved plots/track_a_lr_sensitivity.png")
    return lr_results


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    summary = {
        "landscapes": run_landscape_experiments(),
        "backpropagation_ml": run_backpropagation_and_ml_experiment(),
        "curvature_hvp": run_curvature_and_hvp_experiment(),
        "lr_sensitivity": run_noisy_and_learning_rate_sensitivity()
    }

    with open("results/track_a_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[Track A] Completed successfully! Results written to results/track_a_results.json")
    return summary


if __name__ == "__main__":
    main()
