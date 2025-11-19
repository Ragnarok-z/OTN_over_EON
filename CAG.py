import heapq
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from params import *

class CAG:
    def __init__(self, network, demand, K=3):
        self.network = network
        self.demand = demand
        self.K = K
        self.nodes = set()
        self.edges = defaultdict(dict)  # {u: {v: edge_info}}
        self.build_cag()

    def build_cag(self):
        # Find K shortest paths in G0
        k_shortest_paths = self.network.find_k_shortest_paths(
            self.demand.source, self.demand.destination, self.K)

        # Collect all nodes from these paths that have OTN switching capability
        # self.nodes.add(self.demand.source)
        # self.nodes.add(self.demand.destination)

        for path in k_shortest_paths:
            for node in path:
                if (self.network.otn_switches[node]["used_capacity"] + self.demand.traffic_class.value
                        <= self.network.otn_switches[node]["total_capacity"]):
                    self.nodes.add(node)

        # Add edges between node pairs
        for u in self.nodes:
            for v in self.nodes:
                if u == v:
                    continue

                # Check for existing lightpaths first
                existing_lps = self.network.find_existing_lightpaths(u, v)
                if existing_lps:
                    # Select the best fit (simplified - just take the first one with enough capacity)
                    # completed - selecting the EL that most closely meets the demand’s bandwidth requirements with minimal excess capacity.
                    min_capacity, best_lp = -1, None
                    for lp in existing_lps:
                        if lp.can_accommodate(self.demand):
                            if best_lp is None or lp.remaining_capacity() < min_capacity:
                                min_capacity, best_lp = lp.remaining_capacity(), lp
                    if best_lp:
                        self.edges[u][v] = {"type": "EL", "lightpath": best_lp}
                        continue

                # Check for extendable lightpaths
                extendable_lps = self.network.find_extendable_lightpaths(u, v, self.demand)
                if extendable_lps:
                    # Simplified - just take the first one
                    self.edges[u][v] = {"type": "EEL", "lightpath": extendable_lps[0]}
                    continue

                # Check if we can create a new lightpath
                can_create, mode, path_G0, fs_block = self.network.can_create_lightpath(u, v, self.demand)
                if can_create:
                    self.edges[u][v] = {
                        "type": "PL",
                        "transponder_mode": mode,
                        "path_G0": path_G0,
                        "fs_block": fs_block
                    }

    # 在CAG.py中添加以下方法
    def calculate_abp_fragmentation(self, path_G0):
        """
        计算ABP (Access Blocking Probability) 碎片化指标
        基于论文中的公式：ABP = 1 - (实际容量 / 理想容量)
        """
        # 获取所有可用的传输粒度
        granularities = [8, 12, 19]  # 对应不同的FS需求

        # 计算实际容量（考虑碎片化）


        actual_capacity = 0
        # for u in self.network.edges:
        #     for v in self.network.edges[u]:
        #         if u<v:
        #             edge = (u,v)
        #         else:
        #             continue
        for i in range(len(path_G0) - 1):
            edge = (path_G0[i],path_G0[i+1])
            fs_usage = self.network.fs_usage[edge]

            # 找出所有空闲的连续块
            free_blocks = self._find_free_blocks(fs_usage)

            for block_size in free_blocks:
                for g in granularities:
                    if block_size >= g:
                        actual_capacity += block_size // g

        # 计算理想容量（所有空闲频谱连续）
        total_free_slots = 0
        # for u in self.network.edges:
        #     for v in self.network.edges[u]:
        #         if u<v:
        #             edge = (u,v)
        #         else:
        #             continue
        for i in range(len(path_G0) - 1):
            edge = (path_G0[i],path_G0[i+1])
            total_free_slots += sum(1 for fs in self.network.fs_usage[edge] if not fs)

        ideal_capacity = 0
        for g in granularities:
            if total_free_slots >= g:
                ideal_capacity += total_free_slots // g

        # 避免除零错误
        if ideal_capacity == 0:
            return 0

        abp = 1 - (actual_capacity / ideal_capacity)
        return max(0, min(1, abp))  # 确保在[0,1]范围内

    def _find_free_blocks(self, fs_usage):
        """找出连续的可用频谱块"""
        free_blocks = []
        current_block = 0

        for fs_available in fs_usage:
            if not fs_available:  # 空闲
                current_block += 1
            else:
                if current_block > 0:
                    free_blocks.append(current_block)
                    current_block = 0

        if current_block > 0:
            free_blocks.append(current_block)

        return free_blocks

    def calculate_delta_f(self,path_G0, fs_block):
        """
        计算建立新光路后的碎片化变化量 Δf
        """
        # 计算建立前的ABP
        abp_before = self.calculate_abp_fragmentation(path_G0)

        # 模拟建立光路（临时占用频谱）
        # required_fs = transponder_mode["fs_required"]
        # fs_block = self.network.find_available_fs_block(path_G0, required_fs)

        # if fs_block:
            # 临时分配频谱
        self.network.allocate_fs_block(path_G0, fs_block)

        # 计算建立后的ABP
        abp_after = self.calculate_abp_fragmentation(path_G0)

        # 恢复频谱状态
        self.network.release_fs_block(path_G0, fs_block)

        return abp_after - abp_before

        # return 0

    def calculate_edge_weight(self, u, v, policy):
        edge_info = self.edges[u][v]
        coeffs = POLICY_COEFFICIENTS[policy]

        if edge_info["type"] == "EL" or edge_info["type"] == "EEL":
            lightpath = edge_info["lightpath"]
            if edge_info["type"] == "EL":
                h = len(lightpath.path_in_G0) - 1  # Number of hops
            else:
                h = len(lightpath["extended_lightpath"].path_in_G0) - 1
            return coeffs["c0"] +  (coeffs["c_old"] * h)
        else:  # PL
            mode = edge_info["transponder_mode"]
            path_G0 = edge_info["path_G0"]
            fs_block = edge_info["fs_block"]
            h = len(path_G0) - 1
            # delta_f = 0  # Simplified - actual would calculate fragmentation change
            # 计算实际的Δf（碎片化变化）
            if coeffs["cf"]:
                delta_f = self.calculate_delta_f(path_G0, fs_block)
            else:
                delta_f=0
            u_value = mode["bitrate"]

            weight = (coeffs["c0"] + 1 * (
                    coeffs["c0_new"] +
                    coeffs["c_new"] * h +
                    coeffs["cf"] * delta_f +
                    coeffs["c_prime"] * h ** 2 +
                    coeffs["cu"] * 10 ** (-u_value)
            ))
            return weight

    def find_shortest_path(self, policy, max_hops=5):
        # Implement label-setting algorithm for constrained shortest path
        source = self.demand.source
        destination = self.demand.destination

        # Initialize labels
        labels = {node: [] for node in self.nodes}
        initial_label = {
            "node": source,
            "cost": 0,
            "links": set(),
            "hops": 0,
            "path": [source]
        }
        labels[source].append(initial_label)

        priority_queue = []
        label_id = 0
        heapq.heappush(priority_queue, (initial_label["cost"], label_id, initial_label))
        label_id = 1

        while priority_queue:
            current_cost, _, current_label = heapq.heappop(priority_queue)
            current_node = current_label["node"]

            if current_node == destination:
                return current_label["path"]

            if current_label["hops"] >= max_hops:
                continue

            for neighbor in self.edges[current_node]:
                edge_info = self.edges[current_node][neighbor]
                edge_weight = self.calculate_edge_weight(current_node, neighbor, policy)

                # Create new label
                new_links = current_label["links"].copy()
                if edge_info["type"] in ["PL", "EEL"]:
                    # Add all G0 links to the set
                    path_G0 = edge_info.get("path_G0", [])
                    for i in range(len(path_G0) - 1):
                        new_links.add((path_G0[i], path_G0[i + 1]))

                # Check if any of these links are already used
                if not new_links.isdisjoint(current_label["links"]):
                    continue  # Spectrum conflict

                new_label = {
                    "node": neighbor,
                    "cost": current_label["cost"] + edge_weight,
                    "links": new_links,
                    "hops": current_label["hops"] + 1,
                    "path": current_label["path"] + [neighbor]
                }

                # Check for dominance
                dominated = False
                for existing_label in labels[neighbor]:
                    if (existing_label["cost"] <= new_label["cost"] and
                            existing_label["links"].issubset(new_label["links"]) and
                            existing_label["hops"] <= new_label["hops"]):
                        dominated = True
                        break

                if not dominated:
                    # Add to labels and queue
                    labels[neighbor].append(new_label)
                    heapq.heappush(priority_queue, (new_label["cost"], label_id, new_label))
                    label_id += 1

        return None  # No path found