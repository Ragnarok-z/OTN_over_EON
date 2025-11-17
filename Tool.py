from enum import Enum
import os
import re

# Constants and parameters
class TrafficClass(Enum):
    GE10 = 10
    GE100 = 100
    GE400 = 400

class EventType(Enum):
    ARRIVAL = 1
    DEPARTURE = 2

class Event:
    def __init__(self, event_type, time, demand=None):
        self.event_type = event_type
        self.time = time
        self.demand = demand

    def __lt__(self, other):
        return self.time < other.time


def get_next_exp_number(output_path):
    """获取下一个实验编号"""
    if not os.path.exists(output_path):
        return 0

    # 查找所有已存在的 exp_数字 目录
    existing_dirs = [d for d in os.listdir(output_path)
                     if os.path.isdir(os.path.join(output_path, d)) and re.match(r'exp_\d+', d)]

    if not existing_dirs:
        return 0

    # 提取数字并找到最大值
    numbers = []
    for dir_name in existing_dirs:
        try:
            num = int(dir_name.split('_')[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    if numbers:
        return max(numbers) + 1
    else:
        return 0

import collections

def optimized_spfa(graph, start, end):
    """
    优化的SPFA算法，用于在扩展CAG上寻找最短路径

    Args:
        graph: 邻接表 {u: {v: weight}}
        start: 起点
        end: 终点

    Returns:
        tuple: (最短距离, 路径, 是否找到路径)
    """
    if start not in graph or end not in graph:
        return float('inf'), [], False

    n = max(graph.keys()) + 1 if graph else 0
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    in_queue = {node: False for node in graph}
    count = {node: 0 for node in graph}

    dist[start] = 0
    queue = collections.deque()
    queue.append(start)
    in_queue[start] = True
    count[start] += 1

    # SLF优化参数
    while queue:
        u = queue.popleft()
        in_queue[u] = False

        if u not in graph:
            continue

        for v, weight in graph[u].items():
            if v not in dist:
                continue

            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u

                if not in_queue[v]:
                    count[v] += 1
                    # 检测负环
                    if count[v] >= len(graph):
                        return float('inf'), [], False

                    # SLF优化：如果当前距离小于队首，插入队首
                    if queue and dist[v] < dist[queue[0]]:
                        queue.appendleft(v)
                    else:
                        queue.append(v)
                    in_queue[v] = True

    # 重构路径
    if dist[end] == float('inf'):
        return dist[end], [], False

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()

    return dist[end], path, True