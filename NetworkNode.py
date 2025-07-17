import heapq
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

class NetworkNode:
    def __init__(self, node_id: int, otn_switching_capacity: float):
        self.node_id = node_id
        self.otn_switching_capacity = otn_switching_capacity  # in Gb/s
        self.used_otn_switching = 0.0  # currently used switching capacity