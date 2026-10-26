"""
Workflow Engine - State Storage & Transaction Log (Track J).
Provides atomic key-value state persistence and commit logs.
"""

import threading
from typing import Dict, Any, Optional, List

class StateStore:
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def set(self, key: str, value: Any):
        with self._lock:
            self._store[key] = value
            self._log.append({"action": "SET", "key": key, "val": str(value)[:50]})

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(key, default)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._log.append({"action": "DELETE", "key": key})
                return True
            return False

    def get_commit_log_length(self) -> int:
        with self._lock:
            return len(self._log)
