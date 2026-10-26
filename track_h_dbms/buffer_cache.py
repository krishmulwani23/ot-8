"""
Buffer Pool and Page Cache Replacement Policies (Track H):
Compares LRU, 2Q (Two-Queue), and Clock (Second-Chance) replacement algorithms
under mixed zipfian and sequential scan memory page access traces.
"""

from typing import List, Dict, Set, Any
from collections import OrderedDict
import numpy as np

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, bool] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, page_id: int):
        if page_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(page_id)
        else:
            self.misses += 1
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False) # pop least recently used
            self.cache[page_id] = True


class TwoQueueCache:
    """
    2Q Algorithm (Johnson & Shasha, VLDB 1994):
    Maintains A1in (FIFO queue for new pages) and Am (LRU queue for frequent pages).
    Protects frequently accessed pages against sequential scan pollution!
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.k_in = max(1, capacity // 4) # 25% for first-time pages
        self.k_out = capacity // 2
        self.A1in: OrderedDict[int, bool] = OrderedDict()
        self.A1out: OrderedDict[int, bool] = OrderedDict()
        self.Am: OrderedDict[int, bool] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, page_id: int):
        if page_id in self.Am:
            self.hits += 1
            self.Am.move_to_end(page_id)
        elif page_id in self.A1out:
            self.hits += 1
            # Reclaimed page: promote to long-term queue Am
            del self.A1out[page_id]
            if len(self.Am) + len(self.A1in) >= self.capacity:
                if len(self.Am) > 0:
                    self.Am.popitem(last=False)
            self.Am[page_id] = True
        elif page_id in self.A1in:
            self.hits += 1
            # Still in first-in queue
        else:
            self.misses += 1
            if len(self.A1in) + len(self.Am) >= self.capacity:
                if len(self.A1in) >= self.k_in:
                    # Evict from A1in to ghost cache A1out
                    old_page, _ = self.A1in.popitem(last=False)
                    self.A1out[old_page] = True
                    if len(self.A1out) > self.k_out:
                        self.A1out.popitem(last=False)
                elif len(self.Am) > 0:
                    self.Am.popitem(last=False)
            self.A1in[page_id] = True


class ClockCache:
    """Clock (Second-Chance) page replacement algorithm."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.pages: List[int] = [-1] * capacity
        self.ref_bits: List[int] = [0] * capacity
        self.hand = 0
        self.page_to_slot: Dict[int, int] = {}
        self.hits = 0
        self.misses = 0

    def access(self, page_id: int):
        if page_id in self.page_to_slot:
            self.hits += 1
            slot = self.page_to_slot[page_id]
            self.ref_bits[slot] = 1 # set second-chance bit
        else:
            self.misses += 1
            # Scan clock hand until slot with ref_bit == 0 is found
            while True:
                if self.ref_bits[self.hand] == 0:
                    evicted_page = self.pages[self.hand]
                    if evicted_page in self.page_to_slot:
                        del self.page_to_slot[evicted_page]

                    self.pages[self.hand] = page_id
                    self.ref_bits[self.hand] = 1
                    self.page_to_slot[page_id] = self.hand
                    self.hand = (self.hand + 1) % self.capacity
                    break
                else:
                    self.ref_bits[self.hand] = 0 # clear second chance
                    self.hand = (self.hand + 1) % self.capacity


def benchmark_buffer_pool(page_trace: List[int], capacities: List[int]) -> Dict[str, Any]:
    results = {"capacities": capacities, "LRU": [], "TwoQueue": [], "Clock": []}

    for cap in capacities:
        lru = LRUCache(cap)
        two_q = TwoQueueCache(cap)
        clock = ClockCache(cap)

        for p in page_trace:
            lru.access(p)
            two_q.access(p)
            clock.access(p)

        n = len(page_trace)
        results["LRU"].append(lru.hits / n)
        results["TwoQueue"].append(two_q.hits / n)
        results["Clock"].append(clock.hits / n)

    return results
