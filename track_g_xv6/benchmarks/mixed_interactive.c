// user/mixed_interactive.c - Benchmark workload 3: Mixed interactive burst computation
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int
main(int argc, char *argv[])
{
  int bursts = 20;
  volatile int acc = 0;
  printf("Mixed interactive benchmark started (PID: %d)...\n", getpid());

  for(int b = 0; b < bursts; b++) {
    // Short burst of compute (e.g. processing user keystroke / rendering)
    for(int i = 0; i < 200000; i++) {
      acc += (i % 7);
    }
    // Interactive wait
    sleep(2);
  }

  printf("Mixed interactive finished (PID: %d), acc: %d\n", getpid(), acc);
  exit(0);
}
