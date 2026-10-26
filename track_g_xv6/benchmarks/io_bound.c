// user/io_bound.c - Benchmark workload 2: I/O-intensive task with periodic sleep
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int iterations = 100;
  printf("I/O-bound benchmark started (PID: %d)...\n", getpid());

  for(int i = 0; i < iterations; i++) {
    // Relinquishes CPU before time slice expires
    sleep(1);
  }

  printf("I/O-bound finished (PID: %d)\n", getpid());
  exit(0);
}
