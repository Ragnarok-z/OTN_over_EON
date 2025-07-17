import heapq
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

class EthernetDemand:
    def __init__(self, demand_id: int, source: int, dest: int, traffic_class: str,
                 timeslots_needed: int, holding_time: float,arrival_time: float):
        self.demand_id = demand_id
        self.source = source
        self.dest = dest
        self.traffic_class = traffic_class  # '10GE', '100GE', or '400GE'
        self.timeslots_needed = timeslots_needed
        self.holding_time = holding_time
        self.blocked = False
        self.arrival_time = arrival_time
        self.path = None  # will store the path if not blocked