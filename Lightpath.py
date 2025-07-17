import heapq
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

class Lightpath:
    def __init__(self, source: int, dest: int, transponder_mode: Dict, fs_allocation: Set[int],
                 capacity: float, used_capacity: float = 0.0):
        self.source = source
        self.dest = dest
        self.transponder_mode = transponder_mode  # dict with 'capacity', 'fs_required', 'max_spans'
        self.fs_allocation = fs_allocation  # set of frequency slots used
        self.capacity = capacity  # in Gb/s
        self.used_capacity = used_capacity  # in Gb/s
        self.demands = []  # list of demands carried by this lightpath

    def can_accommodate(self, demand_capacity: float) -> bool:
        return (self.capacity - self.used_capacity) >= demand_capacity

    def add_demand(self, demand_capacity: float):
        self.used_capacity += demand_capacity
        self.demands.append(demand_capacity)