"""
Intermediate Representation (IR) for Mini-Compiler (Track F):
Linear Three-Address Code (TAC), Basic Blocks, and Control Flow Graph (CFG).
"""

from typing import List, Dict, Any, Optional, Set, Tuple

class Instruction:
    """Represents a single Three-Address Code (TAC) instruction: dest = src1 op src2"""
    def __init__(self, op: str, dest: Optional[str] = None,
                 src1: Optional[Any] = None, src2: Optional[Any] = None):
        self.op = op # e.g. 'assign', 'add', 'sub', 'mul', 'div', 'label', 'jmp', 'jmp_false', 'print', 'ret'
        self.dest = dest
        self.src1 = src1
        self.src2 = src2

    def __repr__(self) -> str:
        if self.op == 'label':
            return f"{self.dest}:"
        elif self.op == 'assign':
            return f"  {self.dest} = {self.src1}"
        elif self.op in ('add', 'sub', 'mul', 'div', 'mod', 'lt', 'le', 'eq', 'ne', 'gt', 'ge'):
            op_sym = {
                'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'mod': '%',
                'lt': '<', 'le': '<=', 'eq': '==', 'ne': '!=', 'gt': '>', 'ge': '>='
            }.get(self.op, self.op)
            return f"  {self.dest} = {self.src1} {op_sym} {self.src2}"
        elif self.op == 'jmp':
            return f"  goto {self.dest}"
        elif self.op == 'jmp_false':
            return f"  if not {self.src1} goto {self.dest}"
        elif self.op == 'print':
            return f"  print {self.src1}"
        elif self.op == 'ret':
            return f"  return {self.src1}"
        return f"  {self.op} {self.dest} {self.src1} {self.src2}"

    def defs(self) -> Set[str]:
        """Variables defined (written) by this instruction."""
        if self.dest and self.op not in ('label', 'jmp', 'jmp_false', 'print', 'ret'):
            return {self.dest}
        return set()

    def uses(self) -> Set[str]:
        """Variables used (read) by this instruction."""
        u = set()
        if isinstance(self.src1, str) and not self.src1.lstrip('-').isdigit():
            u.add(self.src1)
        if isinstance(self.src2, str) and not self.src2.lstrip('-').isdigit():
            u.add(self.src2)
        return u


class BasicBlock:
    """Maximal sequence of consecutive instructions with single entry and single exit."""
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.instructions: List[Instruction] = []
        self.predecessors: List['BasicBlock'] = []
        self.successors: List['BasicBlock'] = []
        self.live_in: Set[str] = set()
        self.live_out: Set[str] = set()

    def add_instruction(self, inst: Instruction):
        self.instructions.append(inst)


class ControlFlowGraph:
    """Builds and analyzes Control Flow Graph from TAC instructions."""
    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions
        self.blocks: List[BasicBlock] = []
        self.label_to_block: Dict[str, BasicBlock] = {}
        self.build_cfg()

    def build_cfg(self):
        # 1. Identify leaders
        leaders = set([0])
        labels = {}
        for idx, inst in enumerate(self.instructions):
            if inst.op == 'label':
                leaders.add(idx)
                labels[inst.dest] = idx
            elif inst.op in ('jmp', 'jmp_false'):
                leaders.add(idx + 1)

        # 2. Slice instructions into basic blocks
        sorted_leaders = sorted([i for i in leaders if i < len(self.instructions)])
        block_starts = sorted_leaders + [len(self.instructions)]

        for b_idx in range(len(sorted_leaders)):
            start = block_starts[b_idx]
            end = block_starts[b_idx + 1]
            bb = BasicBlock(b_idx)
            for i in range(start, end):
                bb.add_instruction(self.instructions[i])
                if self.instructions[i].op == 'label':
                    self.label_to_block[self.instructions[i].dest] = bb
            self.blocks.append(bb)
