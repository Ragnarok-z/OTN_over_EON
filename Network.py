import heapq
from collections import defaultdict

from Lightpath import Lightpath
from params import *

import numpy as np
from collections import defaultdict

class Network:
    def __init__(self, topology_file):
        self.nodes = set()
        self.edges = defaultdict(dict)  # G0 layer edges
        self.lightpaths = []  # G1 layer lightpaths
        self.otn_switches = {}  # OTN switching capacity at each node
        self.fs_usage = {}  # Frequency slot usage on each G0 edge
        self.load_topology(topology_file)
        self.initialize_network()

    def load_topology(self, topology_file):
        with open(topology_file, 'r') as f:
            n, m = map(int, f.readline().split())
            for _ in range(m):
                s, t, d = map(int, f.readline().split())
                self.nodes.add(s)
                self.nodes.add(t)
                self.edges[s][t] = d
                self.edges[t][s] = d  # Assuming undirected edges
                # # Initialize FS usage for each edge
                # self.fs_usage[(s, t)] = [False] * 768  # 768 FSs as per the paper
                # self.fs_usage[(t, s)] = [False] * 768
                # 使用numpy布尔数组，更高效
                self.fs_usage[(s, t)] = np.zeros(768, dtype=bool)
                self.fs_usage[(t, s)] = np.zeros(768, dtype=bool)

    def initialize_network(self):
        # Initialize OTN switching capacity (24,000 Gb/s as per the paper)
        for node in self.nodes:
            self.otn_switches[node] = {"total_capacity": 24000, "used_capacity": 0}

    def find_k_shortest_paths(self, source, destination, k):
        # Implement Yen's algorithm to find k shortest paths
        paths = []
        heap = []

        # First find the shortest path
        shortest_path = self.dijkstra(source, destination)
        if not shortest_path:
            return paths
        paths.append(shortest_path)
        heapq.heappush(heap, (self.path_length(shortest_path), 0, shortest_path))

        for i in range(1, k):
            if not heap:
                break

            _, _, prev_path = heapq.heappop(heap)

            for j in range(len(prev_path) - 1):
                spur_node = prev_path[j]
                root_path = prev_path[:j + 1]

                # Remove edges that are part of the root path
                removed_edges = []
                for path in paths:
                    if len(path) > j and root_path == path[:j + 1]:
                        u = path[j]
                        v = path[j + 1] if j + 1 < len(path) else None
                        if v:
                            if v in self.edges[u]:
                                removed_edges.append((u, v, self.edges[u][v]))
                                del self.edges[u][v]
                            if u in self.edges[v]:
                                removed_edges.append((v, u, self.edges[v][u]))
                                del self.edges[v][u]

                # Find spur path from spur node to destination
                spur_path = self.dijkstra(spur_node, destination)

                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    if total_path not in paths:
                        heapq.heappush(heap, (self.path_length(total_path), i, total_path))

                # Restore removed edges
                for u, v, d in removed_edges:
                    self.edges[u][v] = d
                    self.edges[v][u] = d

        # Collect the k shortest paths
        result = []
        while heap and len(result) < k:
            _, _, path = heapq.heappop(heap)
            if path not in result:
                result.append(path)

        return result

    def dijkstra(self, source, destination):
        if source == destination:
            return [source]

        distances = {node: float('inf') for node in self.nodes}
        previous = {node: None for node in self.nodes}
        distances[source] = 0
        heap = [(0, source)]

        while heap:
            current_dist, current_node = heapq.heappop(heap)

            if current_node == destination:
                break

            if current_dist > distances[current_node]:
                continue

            for neighbor, distance in self.edges[current_node].items():
                new_dist = current_dist + distance
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))

        if previous[destination] is None:
            return None

        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        return path[::-1]

    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.edges[path[i]][path[i + 1]]
        return length

    # def find_available_fs_block(self, path, required_fs):
    #     # Find contiguous FS block along the entire path
    #     # Returns (start_fs, end_fs) or None if not available
    #
    #     # Get all edges in the path
    #     edges_in_path = []
    #     for i in range(len(path) - 1):
    #         u = path[i]
    #         v = path[i + 1]
    #         edges_in_path.append((u, v))
    #
    #     # Check each possible FS block
    #     # optimization
    #     start_fs, end_fs = 0, -1
    #     while True:
    #         # print(start_fs,end_fs)
    #         ex_fs = end_fs+1
    #         if ex_fs >= 768:
    #             break
    #         available = True
    #         for edge in edges_in_path:
    #             if self.fs_usage[edge][ex_fs]:
    #                 available = False
    #                 break
    #         if available:
    #             end_fs = ex_fs
    #             if end_fs - start_fs + 1 == required_fs:
    #                 return (start_fs, end_fs)
    #         if not available:
    #             start_fs = ex_fs + 1
    #             end_fs = start_fs - 1
    #             if start_fs + required_fs - 1 >= 768:
    #                 break
    #     return None

    def find_available_fs_block(self, path, required_fs):
        """使用numpy优化查找连续可用频隙块"""
        edges_in_path = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edges_in_path.append((u, v))

        # 使用numpy的位运算来快速查找
        # 计算所有链路的共同可用频隙
        common_available = np.ones(768, dtype=bool)

        for edge in edges_in_path:
            common_available &= ~self.fs_usage[edge]  # 取反表示可用

        # 查找连续可用的频隙块
        if not np.any(common_available):
            return None

        # 使用numpy的运算查找连续块
        diff = np.diff(np.concatenate(([False], common_available, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            if end - start >= required_fs:
                return (start, start + required_fs - 1)

        return None


    def allocate_fs_block(self, path, fs_block):
        start_fs, end_fs = fs_block
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for fs in range(start_fs, end_fs + 1):
                self.fs_usage[(u, v)][fs] = True
                self.fs_usage[(v, u)][fs] = True

    def release_fs_block(self, path, fs_block):
        start_fs, end_fs = fs_block
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for fs in range(start_fs, end_fs + 1):
                self.fs_usage[(u, v)][fs] = False
                self.fs_usage[(v, u)][fs] = False

    def find_existing_lightpaths(self, source, destination):
        # Find existing lightpaths between source and destination with available capacity
        return [lp for lp in self.lightpaths
                if lp.source == source and lp.destination == destination
                and lp.used_capacity < lp.capacity]

    def find_extendable_lightpaths(self, source, destination, demand):
        # return None
        extendable_lps = []
        required_capacity = demand.traffic_class.value

        # 遍历所有现有光路
        for lp in self.lightpaths:
            # 检查是否在源和目的地之间
            if lp.source == source and lp.destination == destination:
                # 检查当前光路是否已经可以容纳需求（如果是，应该作为EL处理，而不是EEL）
                if lp.can_accommodate(demand):
                    continue

                # 获取当前光路的物理路径
                path_G0 = lp.path_in_G0
                path_length = self.path_length(path_G0)
                current_fs_count = lp.fs_allocated[1] - lp.fs_allocated[0] + 1

                # 预先计算左右方向最大可扩展的FS数量（最大不超过19 - current_fs_count）
                max_possible_extension = 19 - current_fs_count
                if max_possible_extension <= 0:
                    continue  # 无法再扩展

                left_available = self.calculate_available_extension(lp, path_G0, "left", max_possible_extension)
                right_available = self.calculate_available_extension(lp, path_G0, "right", max_possible_extension)

                # 寻找合适的扩展模式
                for mode in TRANSPONDER_MODES:
                    # 检查新模式是否能满足容量需求且支持物理距离
                    if (mode["capacity"] >= required_capacity+lp.used_capacity and
                            path_length <= mode["max_spans"] * length_of_span):

                        # 计算需要增加的FS数量
                        required_fs = mode["fs_required"]

                        # 如果新模式需要的FS比当前少，不需要扩展
                        if required_fs <= current_fs_count:
                            continue

                        additional_fs = required_fs - current_fs_count

                        # 检查扩展可行性
                        if additional_fs <= (left_available + right_available):
                            # 优先选择单向扩展（减少频谱碎片）
                            if left_available >= additional_fs:
                                new_start = lp.fs_allocated[0] - additional_fs
                                new_block = (new_start, lp.fs_allocated[1])
                                extendable_lps.append({
                                    "original_lightpath": lp,
                                    "extended_lightpath": Lightpath(source, destination, mode, new_block, path_G0),
                                    "extension_direction": "left",
                                    "additional_fs": additional_fs,
                                    "new_mode": mode
                                })
                                break
                            elif right_available >= additional_fs:
                                new_end = lp.fs_allocated[1] + additional_fs
                                new_block = (lp.fs_allocated[0], new_end)
                                extendable_lps.append({
                                    "original_lightpath": lp,
                                    "extended_lightpath": Lightpath(source, destination, mode, new_block, path_G0),
                                    "extension_direction": "right",
                                    "additional_fs": additional_fs,
                                    "new_mode": mode
                                })
                                break
                            else:
                                # 需要双向扩展
                                # 尽量平衡左右扩展，减少碎片
                                left_extend = min(left_available, additional_fs // 2)
                                right_extend = additional_fs - left_extend

                                # 如果右侧不够，调整分配
                                if right_extend > right_available:
                                    right_extend = right_available
                                    left_extend = additional_fs - right_extend

                                # 再次检查左侧是否足够
                                if left_extend <= left_available:
                                    new_start = lp.fs_allocated[0] - left_extend
                                    new_end = lp.fs_allocated[1] + right_extend
                                    new_block = (new_start, new_end)
                                    extendable_lps.append({
                                        "original_lightpath": lp,
                                        "extended_lightpath": Lightpath(source, destination, mode, new_block, path_G0),
                                        "extension_direction": "both",
                                        "additional_fs": additional_fs,
                                        "new_mode": mode
                                    })
                                    break

        return extendable_lps

    def calculate_available_extension(self, lightpath, path, direction, max_extension):
        """
        计算指定方向最大可扩展的FS数量（不超过max_extension）
        """
        current_block = lightpath.fs_allocated

        # 获取路径中的所有边
        edges_in_path = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edges_in_path.append((u, v))

        available_fs = 0

        if direction == "left":
            # 向左扩展
            start_fs = current_block[0] - 1

            for fs in range(start_fs, start_fs - max_extension, -1):
                if fs < 0:
                    break

                # 检查该FS在所有边上是否可用
                available = True
                for edge in edges_in_path:
                    if self.fs_usage[edge][fs]:
                        available = False
                        break

                if available:
                    available_fs += 1
                else:
                    break

        else:  # direction == "right"
            # 向右扩展
            start_fs = current_block[1] + 1

            for fs in range(start_fs, start_fs + max_extension):
                if fs >= 768:
                    break

                # 检查该FS在所有边上是否可用
                available = True
                for edge in edges_in_path:
                    if self.fs_usage[edge][fs]:
                        available = False
                        break

                if available:
                    available_fs += 1
                else:
                    break

        return available_fs

    def can_create_lightpath(self, source, destination, demand):
        # Check if we can create a new lightpath between source and destination
        # Find the shortest path in G0
        path_G0 = self.dijkstra(source, destination)
        if not path_G0:
            return False

        # Find appropriate transponder mode
        required_capacity = demand.traffic_class.value
        path_length = self.path_length(path_G0)

        # selects the transponder operational mode that minimizes spectral usage while maximizes data rate
        for mode in TRANSPONDER_MODES:
            if mode["capacity"] >= required_capacity and path_length <= mode["max_spans"] * length_of_span:
                # Check FS availability
                fs_block = self.find_available_fs_block(path_G0, mode["fs_required"])
                if fs_block:
                    return True, mode, path_G0, fs_block

        return False, None, None, None

    def create_lightpath(self, source, destination, transponder_mode, fs_block, path_G0):
        # Create a new lightpath
        lightpath = Lightpath(source, destination, transponder_mode, fs_block, path_G0)
        self.lightpaths.append(lightpath)
        self.allocate_fs_block(path_G0, fs_block)
        return lightpath

    def remove_lightpath(self, lightpath):
        if lightpath in self.lightpaths:
            self.lightpaths.remove(lightpath)
            self.release_fs_block(lightpath.path_in_G0, lightpath.fs_allocated)
            return True
        return False

    def update_otn_switching(self, node, capacity_change):
        # Update OTN switching capacity usage at a node
        if node in self.otn_switches:
            self.otn_switches[node]["used_capacity"] += capacity_change
            if self.otn_switches[node]["used_capacity"] < 0:
                self.otn_switches[node]["used_capacity"] = 0

    def get_otn_switching_usage(self, node):
        if node in self.otn_switches:
            return self.otn_switches[node]["used_capacity"]
        return 0