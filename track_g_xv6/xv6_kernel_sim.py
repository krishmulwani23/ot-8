"""
xv6 RISC-V Kernel Execution & Scheduler Trace Simulator (Track G).
Faithfully emulates xv6 process table, timer interrupt handling,
context switching swtch(), and compares Default Round-Robin vs MLFQ Scheduler.
"""

from typing import List, Dict, Any, Optional
import numpy as np

class SimProcess:
    def __init__(self, pid: int, name: str, arrival_tick: int,
                 cpu_bursts: List[int], io_bursts: List[int]):
        self.pid = pid
        self.name = name
        self.arrival_tick = arrival_tick
        self.cpu_bursts = list(cpu_bursts)
        self.io_bursts = list(io_bursts)
        self.state = "RUNNABLE" # RUNNABLE, RUNNING, SLEEPING, ZOMBIE

        # Timing metrics
        self.first_run_tick: Optional[int] = None
        self.completion_tick: Optional[int] = None
        self.total_runtime = 0
        self.wait_time = 0
        self.io_time = 0
        self.num_context_switches = 0

        # MLFQ fields
        self.priority = 0 # 0 (highest), 1, 2 (lowest)
        self.ticks_in_slice = 0

        # Current burst progress
        self.burst_idx = 0
        self.current_cpu_rem = self.cpu_bursts[0] if self.cpu_bursts else 0
        self.current_io_rem = 0

    def is_finished(self) -> bool:
        return self.burst_idx >= len(self.cpu_bursts)


class XV6KernelSimulator:
    def __init__(self, scheduler_type: str = "round_robin", boost_interval: int = 100):
        self.scheduler_type = scheduler_type # 'round_robin' or 'mlfq'
        self.boost_interval = boost_interval
        self.mlfq_slices = [1, 2, 4]
        self.ticks = 0
        self.ticks_since_boost = 0
        self.context_switches = 0
        self.gantt_log: List[Dict[str, Any]] = []

    def run(self, processes: List[SimProcess], max_ticks: int = 500) -> Dict[str, Any]:
        procs = [p for p in processes]
        active_proc: Optional[SimProcess] = None

        while any(not p.is_finished() for p in procs) and self.ticks < max_ticks:
            self.ticks += 1
            self.ticks_since_boost += 1

            # 1. Handle I/O completions for SLEEPING processes
            for p in procs:
                if p.state == "SLEEPING":
                    p.current_io_rem -= 1
                    p.io_time += 1
                    if p.current_io_rem <= 0:
                        p.state = "RUNNABLE"
                        p.burst_idx += 1
                        if p.burst_idx < len(p.cpu_bursts):
                            p.current_cpu_rem = p.cpu_bursts[p.burst_idx]
                        p.ticks_in_slice = 0 # reset slice counter on wakeup

            # 2. Priority boost for MLFQ to prevent starvation
            if self.scheduler_type == "mlfq" and self.ticks_since_boost >= self.boost_interval:
                for p in procs:
                    if not p.is_finished():
                        p.priority = 0
                        p.ticks_in_slice = 0
                self.ticks_since_boost = 0

            # 3. Select next process according to scheduling policy
            runnable = [p for p in procs if p.state == "RUNNABLE" and p.arrival_tick <= self.ticks]

            # If current active process is still running, check if it must be preempted
            if active_proc is not None and active_proc.state == "RUNNING":
                need_preempt = False
                if self.scheduler_type == "round_robin":
                    # 1 tick per slice in standard xv6 RR
                    need_preempt = True
                elif self.scheduler_type == "mlfq":
                    slice_limit = self.mlfq_slices[active_proc.priority]
                    if active_proc.ticks_in_slice >= slice_limit:
                        # Slice expired: demote priority!
                        if active_proc.priority < 2:
                            active_proc.priority += 1
                        active_proc.ticks_in_slice = 0
                        need_preempt = True
                    else:
                        # Check if a strictly higher priority process became runnable
                        if any(p.priority < active_proc.priority for p in runnable):
                            need_preempt = True

                if need_preempt:
                    active_proc.state = "RUNNABLE"
                    runnable.append(active_proc)
                    active_proc = None

            # Choose process to run
            if active_proc is None and runnable:
                if self.scheduler_type == "round_robin":
                    active_proc = runnable[0]
                elif self.scheduler_type == "mlfq":
                    # Choose runnable process with highest priority (lowest number)
                    min_prio = min(p.priority for p in runnable)
                    prio_candidates = [p for p in runnable if p.priority == min_prio]
                    active_proc = prio_candidates[0]

                active_proc.state = "RUNNING"
                active_proc.num_context_switches += 1
                self.context_switches += 1
                if active_proc.first_run_tick is None:
                    active_proc.first_run_tick = self.ticks

            # 4. Advance execution of active process
            if active_proc:
                active_proc.current_cpu_rem -= 1
                active_proc.total_runtime += 1
                active_proc.ticks_in_slice += 1
                self.gantt_log.append({"tick": self.ticks, "pid": active_proc.pid, "name": active_proc.name})

                # Check if current CPU burst is finished
                if active_proc.current_cpu_rem <= 0:
                    active_proc.ticks_in_slice = 0
                    if active_proc.burst_idx < len(active_proc.io_bursts):
                        active_proc.state = "SLEEPING"
                        active_proc.current_io_rem = active_proc.io_bursts[active_proc.burst_idx]
                        active_proc = None
                    else:
                        active_proc.state = "ZOMBIE"
                        active_proc.completion_tick = self.ticks
                        active_proc.burst_idx += 1
                        active_proc = None
            else:
                self.gantt_log.append({"tick": self.ticks, "pid": 0, "name": "IDLE"})

            # 5. Accumulate wait time for all RUNNABLE processes not selected
            for p in procs:
                if p.state == "RUNNABLE" and p != active_proc and p.arrival_tick <= self.ticks:
                    p.wait_time += 1

        # Calculate metrics for each process
        results = {}
        turnarounds = []
        responses = []

        for p in procs:
            turnaround = (p.completion_tick or self.ticks) - p.arrival_tick
            response = (p.first_run_tick or p.arrival_tick) - p.arrival_tick
            turnarounds.append(turnaround)
            responses.append(response)
            results[p.name] = {
                "pid": p.pid,
                "turnaround_time": turnaround,
                "response_time": response,
                "wait_time": p.wait_time,
                "runtime": p.total_runtime,
                "io_time": p.io_time,
                "context_switches": p.num_context_switches,
                "final_priority": p.priority
            }

        # Jain's Fairness Index on throughput / runtime:
        runtimes = [p.total_runtime for p in procs]
        jain_fairness = (sum(runtimes)**2) / (len(runtimes) * sum(r**2 for r in runtimes)) if runtimes else 1.0

        return {
            "scheduler": self.scheduler_type,
            "total_ticks": self.ticks,
            "total_context_switches": self.context_switches,
            "mean_turnaround": float(np.mean(turnarounds)),
            "mean_response": float(np.mean(responses)),
            "jains_fairness": float(jain_fairness),
            "process_stats": results,
            "gantt_log": self.gantt_log
        }
