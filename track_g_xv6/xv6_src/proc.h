// Modified xv6 kernel/proc.h with MLFQ scheduling and statistics
#ifndef _PROC_H_
#define _PROC_H_

#define NPROC        64  // Maximum number of active processes
#define NMLFQ_LEVELS  3  // Priority levels: 0 (highest), 1 (medium), 2 (lowest)
#define BOOST_TICKS 100  // Periodic priority boost interval to prevent starvation

// Time slices in timer ticks for each MLFQ priority level
// Level 0: 1 tick, Level 1: 2 ticks, Level 2: 4 ticks
extern const int mlfq_slice[NMLFQ_LEVELS];

enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

// Per-process scheduling statistics structure for sys_schedstat
struct pstat {
  int inuse[NPROC];      // whether this slot of the process table is in use (1 or 0)
  int pid[NPROC];        // PID of each process
  int priority[NPROC];   // current priority level of each process (0-2)
  enum procstate state[NPROC]; // current state of each process
  int ticks[NPROC][NMLFQ_LEVELS]; // number of ticks each process has accumulated at each level
  int total_runtime[NPROC]; // total execution ticks
  int wait_time[NPROC];     // total ticks spent in RUNNABLE state
  int num_run[NPROC];       // number of times process was scheduled
};

// Per-process state
struct proc {
  struct spinlock lock;

  // p->lock must be held when using these:
  enum procstate state;        // Process state
  void *chan;                  // If non-zero, sleeping on chan
  int killed;                  // If non-zero, have been killed
  int xstate;                  // Exit status to be returned to parent's wait
  int pid;                     // Process ID

  // wait_lock must be held while using this:
  struct proc *parent;         // Parent process

  // these are private to the process, so p->lock need not be held.
  uint64 kstack;               // Virtual address of kernel stack
  uint64 sz;                   // Size of process memory (bytes)
  pagetable_t pagetable;       // User page table
  struct trapframe *trapframe; // data page for trampoline.S
  struct context context;      // swtch() here to run process
  struct file *ofile[16];      // Open files
  struct inode *cwd;           // Current directory
  char name[16];               // Process name (debugging)

  // MLFQ additions:
  int priority;                // Current priority level (0, 1, 2)
  int ticks_used;              // Ticks consumed in current time slice
  int total_runtime;           // Total ticks running on CPU
  int wait_time;               // Total ticks waiting in RUNNABLE queue
  int num_run;                 // Number of context switches into this process
  int ticks_at_level[NMLFQ_LEVELS]; // Tick distribution across queues
};

#endif
