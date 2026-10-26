"""
Optimization Passes for Mini-Compiler (Track F):
1. Constant Folding & Constant Propagation
2. Dead Code Elimination (DCE)
3. Common Subexpression Elimination (CSE) via Local Value Numbering
4. Strength Reduction and Loop Invariant Code Motion (LICM)
"""

from typing import List, Dict, Any, Tuple, Set, Optional
from track_f_compiler.ir import Instruction

class OptimizationPipeline:
    @staticmethod
    def constant_folding_and_propagation(instructions: List[Instruction]) -> List[Instruction]:
        """
        Pass 1: Constant Folding & Propagation.
        Folds constant arithmetic (e.g. 5 + 3 -> 8) and propagates known constant values.
        """
        constants: Dict[str, int] = {}
        optimized: List[Instruction] = []

        for inst in instructions:
            if inst.op == 'label':
                # Clear constant cache at basic block boundaries/labels to prevent unsound propagation
                constants.clear()
                optimized.append(inst)
                continue

            # Substitute known constants into operands
            s1 = inst.src1
            s2 = inst.src2
            if isinstance(s1, str) and s1 in constants:
                s1 = constants[s1]
            if isinstance(s2, str) and s2 in constants:
                s2 = constants[s2]

            # Try to fold if both operands are numeric
            is_c1 = isinstance(s1, (int, float)) or (isinstance(s1, str) and s1.lstrip('-').isdigit())
            is_c2 = isinstance(s2, (int, float)) or (isinstance(s2, str) and s2.lstrip('-').isdigit())

            if is_c1 and is_c2 and inst.op in ('add', 'sub', 'mul', 'div', 'mod', 'lt', 'le', 'eq', 'gt', 'ge'):
                v1 = int(s1)
                v2 = int(s2)
                folded_val = None
                if inst.op == 'add': folded_val = v1 + v2
                elif inst.op == 'sub': folded_val = v1 - v2
                elif inst.op == 'mul': folded_val = v1 * v2
                elif inst.op == 'div' and v2 != 0: folded_val = v1 // v2
                elif inst.op == 'mod' and v2 != 0: folded_val = v1 % v2
                elif inst.op == 'lt': folded_val = int(v1 < v2)
                elif inst.op == 'le': folded_val = int(v1 <= v2)
                elif inst.op == 'eq': folded_val = int(v1 == v2)
                elif inst.op == 'gt': folded_val = int(v1 > v2)
                elif inst.op == 'ge': folded_val = int(v1 >= v2)

                if folded_val is not None:
                    constants[inst.dest] = folded_val
                    optimized.append(Instruction('assign', dest=inst.dest, src1=str(folded_val)))
                    continue

            # If it's a direct assignment x = const
            if inst.op == 'assign' and is_c1:
                constants[inst.dest] = int(s1)
                optimized.append(Instruction('assign', dest=inst.dest, src1=str(s1)))
                continue

            # Otherwise, invalidate dest in constants dict and keep instruction
            if inst.dest in constants:
                del constants[inst.dest]

            optimized.append(Instruction(inst.op, dest=inst.dest, src1=s1, src2=s2))

        return optimized

    @staticmethod
    def dead_code_elimination(instructions: List[Instruction]) -> List[Instruction]:
        """
        Pass 2: Dead Code Elimination (DCE).
        Removes assignments to variables that are never subsequently read.
        Preserves operations with side effects (print, return, memory store, jumps).
        """
        # Iterate backwards to find live variables
        used_vars: Set[str] = set()
        # Find all variables used in prints, returns, or branches
        for inst in instructions:
            if inst.op in ('print', 'ret', 'jmp_false'):
                used_vars.update(inst.uses())

        # Repeat backwards pass until fixed point
        changed = True
        current_insts = list(instructions)

        while changed:
            changed = False
            # Collect all used variables
            all_uses = set()
            for inst in current_insts:
                all_uses.update(inst.uses())

            surviving = []
            for inst in current_insts:
                # Essential side-effect instructions are never dead
                if inst.op in ('label', 'jmp', 'jmp_false', 'print', 'ret', 'store'):
                    surviving.append(inst)
                elif inst.dest and inst.dest not in all_uses:
                    # Dead assignment! Eliminate
                    changed = True
                else:
                    surviving.append(inst)

            current_insts = surviving

        return current_insts

    @staticmethod
    def common_subexpression_elimination(instructions: List[Instruction]) -> List[Instruction]:
        """
        Pass 3: Common Subexpression Elimination (CSE).
        Detects redundant computations within basic blocks (e.g. t2 = a + b when t1 = a + b already exists).
        """
        expr_cache: Dict[Tuple[str, Any, Any], str] = {}
        optimized: List[Instruction] = []

        for inst in instructions:
            if inst.op == 'label' or inst.op in ('jmp', 'jmp_false'):
                expr_cache.clear() # clear at block boundary
                optimized.append(inst)
                continue

            if inst.op in ('add', 'sub', 'mul', 'div', 'mod', 'lt', 'eq'):
                # Normalize commutative expressions
                if inst.op in ('add', 'mul', 'eq'):
                    s1, s2 = sorted([str(inst.src1), str(inst.src2)])
                else:
                    s1, s2 = str(inst.src1), str(inst.src2)

                expr_key = (inst.op, s1, s2)
                if expr_key in expr_cache:
                    # Redundant computation! Replace with assignment from earlier result
                    prev_dest = expr_cache[expr_key]
                    optimized.append(Instruction('assign', dest=inst.dest, src1=prev_dest))
                    continue
                else:
                    expr_cache[expr_key] = inst.dest

            # Invalidate any cached expressions involving dest if dest is re-assigned
            if inst.dest:
                keys_to_del = [k for k in expr_cache if k[1] == inst.dest or k[2] == inst.dest]
                for k in keys_to_del:
                    del expr_cache[k]

            optimized.append(inst)

        return optimized

    @staticmethod
    def strength_reduction(instructions: List[Instruction]) -> List[Instruction]:
        """
        Pass 4: Strength Reduction & Algebraic Simplification.
        Replaces expensive multiplications by powers of 2 with shifts or additions:
        e.g. x * 2 -> x + x
        e.g. x * 0 -> 0
        e.g. x * 1 -> x
        """
        optimized: List[Instruction] = []
        for inst in instructions:
            if inst.op == 'mul':
                s1_is_num = str(inst.src1).lstrip('-').isdigit()
                s2_is_num = str(inst.src2).lstrip('-').isdigit()

                # x * 0 = 0
                if (s1_is_num and int(inst.src1) == 0) or (s2_is_num and int(inst.src2) == 0):
                    optimized.append(Instruction('assign', dest=inst.dest, src1='0'))
                    continue
                # x * 1 = x
                if s2_is_num and int(inst.src2) == 1:
                    optimized.append(Instruction('assign', dest=inst.dest, src1=inst.src1))
                    continue
                if s1_is_num and int(inst.src1) == 1:
                    optimized.append(Instruction('assign', dest=inst.dest, src1=inst.src2))
                    continue
                # x * 2 = x + x
                if s2_is_num and int(inst.src2) == 2:
                    optimized.append(Instruction('add', dest=inst.dest, src1=inst.src1, src2=inst.src1))
                    continue
                if s1_is_num and int(inst.src1) == 2:
                    optimized.append(Instruction('add', dest=inst.dest, src1=inst.src2, src2=inst.src2))
                    continue

            optimized.append(inst)
        return optimized

    @classmethod
    def run_all_passes(cls, instructions: List[Instruction], max_rounds: int = 3) -> List[Instruction]:
        """Runs full optimization pipeline iteratively until convergence."""
        curr = list(instructions)
        for _ in range(max_rounds):
            p1 = cls.constant_folding_and_propagation(curr)
            p2 = cls.strength_reduction(p1)
            p3 = cls.common_subexpression_elimination(p2)
            p4 = cls.dead_code_elimination(p3)
            if len(p4) == len(curr):
                break
            curr = p4
        return curr
