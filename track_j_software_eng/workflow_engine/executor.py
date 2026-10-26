"""
Workflow Engine - Parallel Execution Pool & Scheduler (Track J).
Dispatches tasks concurrently while enforcing DAG dependencies.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Set
from track_j_software_eng.workflow_engine.dag import DAG
from track_j_software_eng.workflow_engine.task import Task, TaskState

class WorkflowExecutor:
    """Coordinates parallel execution of tasks according to DAG precedence constraints."""
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.lock = threading.Lock()

    def run_workflow(self, dag: DAG, tasks: Dict[str, Task]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        completed_nodes: Set[str] = set()
        failed_nodes: Set[str] = set()
        results: Dict[str, Any] = {}
        all_nodes = set(dag.nodes.keys())

        # Thread pool for concurrent execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}

            while len(completed_nodes) + len(failed_nodes) < len(all_nodes):
                with self.lock:
                    # Find eligible nodes whose parent dependencies are all completed
                    ready_nodes = []
                    for nid in all_nodes - completed_nodes - failed_nodes:
                        if nid not in [f_nid for f_nid in futures.values()]:
                            node = dag.nodes[nid]
                            if node.dependencies.issubset(completed_nodes):
                                ready_nodes.append(nid)

                    # Submit ready nodes to worker pool
                    for nid in ready_nodes:
                        task = tasks.get(nid)
                        if task:
                            fut = pool.submit(task.execute)
                            futures[fut] = nid

                if not futures:
                    if len(completed_nodes) + len(failed_nodes) < len(all_nodes):
                        # Deadlock or unfulfillable dependencies
                        break
                    break

                # Wait for at least one task future to complete
                done_futures = []
                for fut in as_completed(futures):
                    done_futures.append(fut)
                    break # process as soon as one finishes

                for fut in done_futures:
                    nid = futures.pop(fut)
                    try:
                        res = fut.result()
                        results[nid] = res
                        completed_nodes.add(nid)
                    except Exception as err:
                        failed_nodes.add(nid)
                        results[nid] = {"error": str(err)}

        total_runtime = time.perf_counter() - t0
        return {
            "completed_count": len(completed_nodes),
            "failed_count": len(failed_nodes),
            "total_runtime_s": total_runtime,
            "results": results
        }
