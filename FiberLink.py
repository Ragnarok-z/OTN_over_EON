import heapq
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

class FiberLink:
    def __init__(self, source: int, dest: int, length: float, total_fs: int = 768):
        self.source = source
        self.dest = dest
        self.length = length  # in km
        self.total_fs = total_fs
        self.used_fs = set()  # set of used frequency slots
        self.available_fs = set(range(total_fs))  # initially all FS are available