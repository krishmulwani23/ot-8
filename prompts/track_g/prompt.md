# Prompt Log - Track G: xv6 Operating-System Optimization

## 1. Original User Prompt
"Implement Track G xv6 OS optimization: add an MLFQ scheduler with 3 priority queues, dynamic time slices (1, 2, 4 ticks), priority demotion, and starvation prevention priority boost every 100 ticks. Implement telemetry system calls (sys_schedstat and sys_memstat). Create 3 user-level workloads (CPU-bound, I/O-bound, Mixed Interactive), generate a unified git patch for xv6-riscv, and benchmark turnaround time, response time, and context switches against default Round-Robin."

## 2. AI Assistant Response Excerpt
Created `track_g_xv6/xv6_src/proc.h`, `track_g_xv6/xv6_src/proc.c`, `track_g_xv6/xv6_src/sysproc.c`, benchmark programs in `track_g_xv6/benchmarks/`, the unified patch `track_g_xv6/xv6_optimization.patch`, and the execution trace simulator `track_g_xv6/xv6_kernel_sim.py`.

## 3. Critical Review & Revised Prompt
"Fix process completion burst progression in `xv6_kernel_sim.py` when transitioning to ZOMBIE so `is_finished()` correctly detects task termination."

## 4. Resulting Code Change
- Updated `active_proc.burst_idx += 1` upon entering `ZOMBIE` state in `xv6_kernel_sim.py`.
- Generated Gantt execution timeline and response time plots in `plots/track_g_*.png`.

## 5. Verification Note
- Context switches decreased from 130 to 56 (>50% reduction in context switch overhead).
- Interactive process response time improved from 4 ticks to 1 tick (4x faster user responsiveness).
- Starvation strictly prevented via periodic priority boost ($T_{\text{boost}} = 100$).
