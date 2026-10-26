// user/cpu_bound.c - Benchmark workload 1: CPU-intensive calculation
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int n = 50000000;
  volatile double x = 0.0;
  printf("CPU-bound benchmark started (PID: %d)...\n", getpid());

  for(int i = 0; i < n; i++) {
    x += (i * 0.001) / (i + 1.0);
  }

  printf("CPU-bound finished (PID: %d), res: %d\n", getpid(), (int)x);
  exit(0);
}
