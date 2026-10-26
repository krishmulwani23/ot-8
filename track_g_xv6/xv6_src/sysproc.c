// Modified xv6 kernel/sysproc.c with sys_schedstat and sys_memstat system calls
#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"

// System call to retrieve scheduling telemetry
uint64
sys_schedstat(void)
{
  uint64 dst_addr;
  argaddr(0, &dst_addr);

  struct pstat stat;
  struct proc *p;
  int i = 0;

  for(p = proc; p < &proc[NPROC]; p++, i++) {
    acquire(&p->lock);
    stat.inuse[i] = (p->state != UNUSED);
    stat.pid[i] = p->pid;
    stat.priority[i] = p->priority;
    stat.state[i] = p->state;
    stat.total_runtime[i] = p->total_runtime;
    stat.wait_time[i] = p->wait_time;
    stat.num_run[i] = p->num_run;
    for(int l = 0; l < NMLFQ_LEVELS; l++) {
      stat.ticks[i][l] = p->ticks_at_level[l];
    }
    release(&p->lock);
  }

  // Copy data to user space
  if(copyout(myproc()->pagetable, dst_addr, (char*)&stat, sizeof(stat)) < 0)
    return -1;
  return 0;
}

// System call to report memory and buffer cache statistics
uint64
sys_memstat(void)
{
  uint64 dst_addr;
  argaddr(0, &dst_addr);

  struct {
    uint64 free_bytes;
    uint64 total_bytes;
    uint64 bcache_hits;
    uint64 bcache_misses;
  } mstat;

  mstat.free_bytes = kalloc_free_bytes();
  mstat.total_bytes = PHYSTOP - KERNBASE;
  mstat.bcache_hits = bcache_get_hits();
  mstat.bcache_misses = bcache_get_misses();

  if(copyout(myproc()->pagetable, dst_addr, (char*)&mstat, sizeof(mstat)) < 0)
    return -1;
  return 0;
}
