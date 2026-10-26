"""
Virtual Machine / TAC Interpreter (Track F):
Executes Three-Address Code, tracks runtime cycles, and verifies semantic equivalence.
"""

from typing import List, Dict, Any, Tuple
from track_f_compiler.ir import Instruction

class TACVirtualMachine:
    """Simulates CPU execution of Three-Address Code."""
    def __init__(self, memory_size: int = 1024):
        self.memory = [0] * memory_size
        self.registers: Dict[str, int] = {}
        self.output: List[str] = []
        self.cycles: int = 0

    def reset(self):
        self.registers.clear()
        self.output.clear()
        self.cycles = 0

    def _resolve(self, val: Any) -> int:
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            if val.lstrip('-').isdigit():
                return int(val)
            return self.registers.get(val, 0)
        return 0

    def execute(self, instructions: List[Instruction], max_cycles: int = 100000) -> Dict[str, Any]:
        self.reset()
        # Build label index
        labels: Dict[str, int] = {}
        for idx, inst in enumerate(instructions):
            if inst.op == 'label':
                labels[inst.dest] = idx

        pc = 0
        n_inst = len(instructions)

        while pc < n_inst and self.cycles < max_cycles:
            inst = instructions[pc]
            self.cycles += 1
            op = inst.op

            if op == 'label':
                pc += 1
            elif op == 'assign':
                self.registers[inst.dest] = self._resolve(inst.src1)
                pc += 1
            elif op in ('add', 'sub', 'mul', 'div', 'mod', 'lt', 'le', 'eq', 'ne', 'gt', 'ge'):
                v1 = self._resolve(inst.src1)
                v2 = self._resolve(inst.src2)
                res = 0
                if op == 'add': res = v1 + v2
                elif op == 'sub': res = v1 - v2
                elif op == 'mul': res = v1 * v2
                elif op == 'div': res = v1 // v2 if v2 != 0 else 0
                elif op == 'mod': res = v1 % v2 if v2 != 0 else 0
                elif op == 'lt': res = int(v1 < v2)
                elif op == 'le': res = int(v1 <= v2)
                elif op == 'eq': res = int(v1 == v2)
                elif op == 'ne': res = int(v1 != v2)
                elif op == 'gt': res = int(v1 > v2)
                elif op == 'ge': res = int(v1 >= v2)
                self.registers[inst.dest] = res
                pc += 1
            elif op == 'jmp':
                pc = labels.get(inst.dest, pc + 1)
            elif op == 'jmp_false':
                cond = self._resolve(inst.src1)
                if not cond:
                    pc = labels.get(inst.dest, pc + 1)
                else:
                    pc += 1
            elif op == 'print':
                val = self._resolve(inst.src1)
                self.output.append(str(val))
                pc += 1
            elif op == 'ret':
                break
            else:
                pc += 1

        return {
            "output": list(self.output),
            "cycles": self.cycles,
            "final_registers": dict(self.registers),
            "terminated_cleanly": (pc >= n_inst or (pc < n_inst and instructions[pc].op == 'ret'))
        }
