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


def optimized_spfa_with_g0_constraint(cag, start, end):
    """
    带G0层链路限制的优化SPFA算法

    Args:
        cag: ExtendedCAG实例
        start: 起点
        end: 终点

    Returns:
        tuple: (最短距离, 路径, 使用的边列表, 是否找到路径)
    """
    if start not in cag.nodes or end not in cag.nodes:
        return float('inf'), [], [], False

    # 初始化数据结构
    dist = {node: float('inf') for node in cag.nodes}
    prev = {node: None for node in cag.nodes}  # 记录前驱节点
    prev_edge = {node: None for node in cag.nodes}  # 记录前驱边
    used_g0_links = {node: set() for node in cag.nodes}  # 记录到每个节点已使用的G0链路
    in_queue = {node: False for node in cag.nodes}
    count = {node: 0 for node in cag.nodes}

    dist[start] = 0
    used_g0_links[start] = set()
    queue = collections.deque()
    queue.append(start)
    in_queue[start] = True
    count[start] += 1

    while queue:
        u = queue.popleft()
        in_queue[u] = False

        # 遍历所有相邻节点
        for v in cag.edges[u]:
            # 遍历u到v的所有边
            for edge_info in cag.edges[u][v]:
                weight = edge_info["weight"]
                new_dist = dist[u] + weight

                # 检查G0链路限制
                if edge_info["type"] in ["EEL", "PL"]:
                    edge_g0_links = edge_info["g0_links"]
                    # 检查是否有G0链路冲突
                    if used_g0_links[u] & edge_g0_links:
                        continue  # 有冲突，跳过这条边

                # 如果距离更短，更新
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    prev_edge[v] = edge_info

                    # 更新已使用的G0链路
                    if edge_info["type"] in ["EEL", "PL"]:
                        used_g0_links[v] = used_g0_links[u] | edge_info["g0_links"]
                    else:
                        used_g0_links[v] = used_g0_links[u].copy()

                    if not in_queue[v]:
                        count[v] += 1
                        # 检测负环
                        if count[v] >= len(cag.nodes):
                            return float('inf'), [], [], False

                        # SLF优化
                        if queue and dist[v] < dist[queue[0]]:
                            queue.appendleft(v)
                        else:
                            queue.append(v)
                        in_queue[v] = True

    # 重构路径和边序列
    if dist[end] == float('inf'):
        return dist[end], [], [], False

    path = []
    edge_sequence = []
    current = end

    while current is not None:
        path.append(current)
        if prev_edge[current] is not None:
            edge_sequence.append(prev_edge[current])
        current = prev[current]

    path.reverse()
    edge_sequence.reverse()

    assert None not in path, f"path={path}"
    assert None not in edge_sequence, f"path={edge_sequence}"

    return dist[end], path, edge_sequence, True