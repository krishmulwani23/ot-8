"""
Workflow Engine - Task State Machine & Retry Policies (Track J).
Implements execution states, exponential backoff, jitter, and error handling.
"""

import time
import math
from enum import Enum
from typing import Callable, Any, Optional, Dict

class TaskState(Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    SKIPPED = "SKIPPED"


class RetryPolicy:
    """Configures retry behavior with exponential backoff and jitter."""
    def __init__(self, max_retries: int = 3, initial_delay: float = 0.1,
                 backoff_multiplier: float = 2.0, max_delay: float = 5.0):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay

    def get_delay(self, attempt: int) -> float:
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_delay)


class Task:
    """Executable task encapsulation."""
    def __init__(self, task_id: str, action: Callable[..., Any],
                 retry_policy: Optional[RetryPolicy] = None, timeout: float = 30.0):
        self.task_id = task_id
        self.action = action
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout = timeout
        self.state = TaskState.PENDING

        self.attempts = 0
        self.result: Any = None
        self.error: Optional[Exception] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def execute(self, *args, **kwargs) -> Any:
        self.attempts += 1
        self.start_time = time.perf_counter()
        self.state = TaskState.RUNNING

        try:
            self.result = self.action(*args, **kwargs)
            self.state = TaskState.SUCCESS
            self.end_time = time.perf_counter()
            return self.result
        except Exception as exc:
            self.error = exc
            self.end_time = time.perf_counter()
            if self.attempts <= self.retry_policy.max_retries:
                self.state = TaskState.RETRYING
            else:
                self.state = TaskState.FAILED
            raise exc

    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
