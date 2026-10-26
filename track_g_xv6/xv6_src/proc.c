// Modified xv6 kernel/proc.c with MLFQ Scheduler implementation
#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"

struct cpu cpus[NCPU];
struct proc proc[NPROC];
struct proc *initproc;

int nextpid = 1;
struct spinlock pid_lock;

const int mlfq_slice[NMLFQ_LEVELS] = { 1, 2, 4 };
static int total_ticks_since_boost = 0;

// Priority boost function to prevent starvation of low-priority CPU-bound processes
void
priority_boost(void)
{
  struct proc *p;
  for(p = proc; p < &proc[NPROC]; p++) {
    acquire(&p->lock);
    if(p->state != UNUSED) {
      p->priority = 0;
      p->ticks_used = 0;
    }
    release(&p->lock);
  }
}

// Multi-Level Feedback Queue Scheduler
// Replaces default Round-Robin scheduler
void
scheduler(void)
{
  struct proc *p;
  struct cpu *c = mycpu();
  c->proc = 0;

  for(;;){
    // Enable interrupts on this processor to avoid deadlock
    intr_on();

    // Check for periodic priority boost
    if(total_ticks_since_boost >= BOOST_TICKS) {
      priority_boost();
      total_ticks_since_boost = 0;
    }

    struct proc *best_proc = 0;

    // MLFQ Selection: scan priority levels 0, 1, 2 in order of precedence
    for(int level = 0; level < NMLFQ_LEVELS; level++) {
      for(p = proc; p < &proc[NPROC]; p++) {
        acquire(&p->lock);
        if(p->state == RUNNABLE && p->priority == level) {
          best_proc = p;
          break; // Found highest priority runnable process
        }
        release(&p->lock);
      }
      if(best_proc)
        break;
    }

    if(best_proc) {
      // Run the chosen process
      best_proc->state = RUNNING;
      best_proc->num_run++;
      c->proc = best_proc;
      swtch(&c->context, &best_proc->context);

      // Process has yielded or timer interrupt occurred
      c->proc = 0;
      release(&best_proc->lock);
    }
  }
}

// Called from trap.c on timer interrupt
void
update_proc_ticks(void)
{
  struct proc *p = myproc();
  total_ticks_since_boost++;

  // Increment wait time for all other RUNNABLE processes
  for(struct proc *other = proc; other < &proc[NPROC]; other++) {
    if(other != p && other->state == RUNNABLE) {
      other->wait_time++;
    }
  }

  if(p && p->state == RUNNING) {
    p->ticks_used++;
    p->total_runtime++;
    p->ticks_at_level[p->priority]++;

    // If process exhausted its time slice at current priority level:
    // Demote priority (Rule 4a of MLFQ)
    if(p->ticks_used >= mlfq_slice[p->priority]) {
      if(p->priority < NMLFQ_LEVELS - 1) {
        p->priority++;
      }
      p->ticks_used = 0;
      yield(); // Relinquish CPU
    }
  }
}
