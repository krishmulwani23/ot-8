# Prompt Log - Track A: Continuous Optimization

## 1. Original User Prompt
"Implement all continuous optimization methods from Chapter 6: Gradient Descent, SGD, Mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, and second-order methods (Newton-Raphson, Gauss-Newton, curvature approximations: empirical Fisher, diagonal Hessian, Hessian-vector products). Show backpropagation as distinct from the optimizer on a small neural net, test across convex and non-convex landscapes (Rosenbrock, saddle points, Rastrigin), and evaluate step-size tuning and noise robustness."

## 2. AI Assistant Response Excerpt
Generated `track_a_continuous/landscapes.py`, `track_a_continuous/optimizers.py`, `track_a_continuous/second_order.py`, and `track_a_continuous/backprop_model.py`.
Separated gradient computation (`TwoLayerMLP.backpropagation`) from the optimizer parameter update rule (`Adam.step`, `Momentum.step`), and provided exact HVP via two-point finite difference of gradients.

## 3. Critical Review & Revised Prompt
"Ensure backpropagation is strictly decoupled from weight updates and verified against numerical finite-difference gradients. Add Hessian-free Newton-CG step using purely Hessian-Vector Products."

## 4. Resulting Code Change
- Implemented `TwoLayerMLP.verify_gradients_numerical` asserting relative error $< 10^{-4}$.
- Added `CurvatureApproximations.newton_cg_step` solving $H \Delta x = g$ without materializing the $d \times d$ Hessian matrix.

## 5. Verification Note
- Numerical backpropagation gradient verification achieved relative error: $3.657 \times 10^{-11}$ (PASS).
- HVP approximation error: $1.049 \times 10^{-10}$ (PASS).
- Convergence plots generated in `plots/track_a_convergence.png` and `plots/track_a_lr_sensitivity.png`.
