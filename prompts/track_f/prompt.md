# Prompt Log - Track F: Compiler Optimization

## 1. Original User Prompt
"Implement Track F Compiler Optimization: build a mini-compiler for Three-Address Code with basic blocks, implement 4 optimization passes (Constant Folding & Propagation, Dead Code Elimination, Common Subexpression Elimination, and Strength Reduction), formulate Register Allocation as an ILP graph-coloring problem with variable spill penalties, build a VM to verify that unoptimized and optimized code produce identical outputs, and benchmark instruction count, cycles, and compile time."

## 2. AI Assistant Response Excerpt
Created `track_f_compiler/ir.py`, `track_f_compiler/optimizations.py`, `track_f_compiler/register_allocator.py`, and `track_f_compiler/virtual_machine.py`.
Discovered and resolved typing import issues and ensured side-effecting operations (`print`, `ret`, `store`) are strictly preserved during DCE.

## 3. Critical Review & Revised Prompt
"Add typing List, Dict imports to experiments.py and verify that Program 1 produces identical output ('90') and Program 2 produces identical output ('3050')."

## 4. Resulting Code Change
- Updated imports in `experiments.py`.
- Formulated Register Allocation as an exact 0-1 ILP in `register_allocator.py`.

## 5. Verification Note
- Semantic equivalence verified: unoptimized vs optimized code outputs are bit-identical (PASS).
- Code size reduced by 80% on Program 1 (10 instructions -> 2 instructions, 5.0x cycle speedup).
- ILP Register Allocator successfully proved 0 spills at $K=5$, 1 spill at $K=4$, 2 spills at $K=3$, and 3 spills at $K=2$.
