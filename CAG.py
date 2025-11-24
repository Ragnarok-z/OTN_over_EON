import heapq
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from params import *

class CAG:
    def __init__(self, network, demand, K=3, include_OTN_frag=False):
        self.network = network
        self.demand = demand
        self.K = K
        self.include_OTN_frag = include_OTN_frag  # 添加新参数
        self.nodes = set()
        self.edges = defaultdict(dict)  # {u: {v: edge_info}}
        self.build_cag(include_OTN_frag)

    def build_cag(self, include_OTN_frag=False):
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
                    if include_OTN_frag:
                        # 使用新策略：select_EL函数
                        best_lp = self.select_EL([lp for lp in existing_lps if lp.can_accommodate(self.demand)])
                    else:
                        # 使用原策略
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
                    if include_OTN_frag:
                        # 使用新策略：select_EEL函数
                        best_eel = self.select_EEL(extendable_lps)
                    else:
                        # 使用原策略：取第一个
                        best_eel = extendable_lps[0] if extendable_lps else None

                    if best_eel:
                        self.edges[u][v] = {"type": "EEL", "lightpath": best_eel}
                        continue

                # Check if we can create a new lightpath
                if include_OTN_frag:
                    # 使用新策略：find_PL + select_PL
                    pl_list = self.find_PL(u, v, self.demand, k_pl=3)
                    best_pl = self.select_PL(pl_list)
                    if best_pl:
                        self.edges[u][v] = {
                            "type": "PL",
                            "transponder_mode": best_pl["transponder_mode"],
                            "path_G0": best_pl["path_G0"],
                            "fs_block": best_pl["fs_block"]
                        }
                else:
                    # 使用原策略
                    can_create, mode, path_G0, fs_block = self.network.can_create_lightpath(u, v, self.demand)
                    if can_create:
                        self.edges[u][v] = {
                            "type": "PL",
                            "transponder_mode": mode,
                            "path_G0": path_G0,
                            "fs_block": fs_block
                        }

    # 在 CAG 类中添加以下方法

    def select_EL(self, EL_lst):
        """
        从平行EL列表中选择一个EL
        选择标准：
        1. 优先选择占用G0链路数最少的EL
        2. 如果G0链路数相同，选择剩余容量最少的EL（最接近需求容量的）
        """
        if not EL_lst:
            return None

        # 如果只有一个EL，直接返回
        if len(EL_lst) == 1:
            return EL_lst[0]

        # 计算每个EL的G0链路数和剩余容量
        el_info = []
        for i,el in enumerate(EL_lst):
            # 计算G0链路数（物理路径的跳数）
            g0_link_count = len(el.path_in_G0) - 1

            # 计算剩余容量
            remaining_capacity = el.remaining_capacity()

            el_info.append({
                'id': i if i>0 else '000000000000',
                'el': el,
                'g0_link_count': g0_link_count,
                'remaining_capacity': remaining_capacity
            })

        # 按照选择标准排序：
        # 1. 优先G0链路数少的（升序）
        # 2. 然后剩余容量少的（升序）
        sorted_els = sorted(el_info,
                            key=lambda x: (x['g0_link_count'], x['remaining_capacity']))

        # print(len(EL_lst), sorted_els[0]['id'])

        # 返回最优的EL
        return sorted_els[1]['el']

    def select_EEL(self, extendable_lps):
        """
        从可扩展光路列表中选择一个EEL
        选择标准：
        1. 优先选择剩余容量最少的EEL（扩展后光路的剩余容量）
        2. 如果剩余容量相同，选择G0中起始频隙最靠前的
        """
        if not extendable_lps:
            return None

        # 如果只有一个EEL，直接返回
        if len(extendable_lps) == 1:
            return extendable_lps[0]

        # 计算每个EEL的剩余容量和起始频隙
        eel_info = []
        for eel in extendable_lps:
            # 获取扩展后的光路
            extended_lightpath = eel["extended_lightpath"]

            # 计算扩展后光路的剩余容量
            # 注意：这里计算的是扩展后光路容纳当前需求后的剩余容量
            demand_capacity = self.demand.traffic_class.value
            remaining_capacity = extended_lightpath.capacity - (
                        eel["original_lightpath"].used_capacity + demand_capacity)

            # 获取起始频隙
            start_fs = extended_lightpath.fs_allocated[0]

            eel_info.append({
                'eel': eel,
                'remaining_capacity': remaining_capacity,
                'start_fs': start_fs
            })

        # 按照选择标准排序：
        # 1. 优先剩余容量少的（升序）
        # 2. 然后起始频隙小的（升序）
        sorted_eels = sorted(eel_info,
                             key=lambda x: (x['remaining_capacity'], x['start_fs']))

        # 返回最优的EEL
        return sorted_eels[0]['eel']

    def find_PL(self, u, v, demand, k_pl=3):
        """
        基于shortest-path, first-k-fit方案寻找k_pl个PL
        返回前k个可建立的PL配置
        """
        # 找到最短路径
        path_G0 = self.network.dijkstra(u, v)
        if not path_G0:
            return []

        required_capacity = demand.traffic_class.value
        path_length = self.network.path_length(path_G0)

        available_pls = []

        # 检查所有可能的传输机模式
        for mode in TRANSPONDER_MODES:
            if mode["capacity"] >= required_capacity and path_length <= mode["max_spans"] * length_of_span:
                # 使用first-k-fit策略找到多个可用的频谱块
                fs_blocks = self.find_k_available_fs_blocks(path_G0, mode["fs_required"], k_pl)

                for fs_block in fs_blocks:
                    if fs_block:
                        available_pls.append({
                            "transponder_mode": mode,
                            "path_G0": path_G0,
                            "fs_block": fs_block
                        })

                        # 如果已经找到k_pl个，就停止
                        if len(available_pls) >= k_pl:
                            break

        return available_pls

    def find_k_available_fs_blocks(self, path, required_fs, k):
        """
        找到前k个可用的连续频谱块
        """
        edges_in_path = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edges_in_path.append((u, v))

        # 使用numpy的位运算来快速查找
        common_available = np.ones(768, dtype=bool)
        for edge in edges_in_path:
            common_available &= ~self.network.fs_usage[edge]

        if not np.any(common_available):
            return []

        # 查找所有连续可用的频隙块
        diff = np.diff(np.concatenate(([False], common_available, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        available_blocks = []
        for start, end in zip(starts, ends):
            block_size = end - start
            if block_size >= required_fs:
                # 这个连续块中可以容纳多个频谱块
                num_blocks_in_segment = block_size - required_fs + 1
                for i in range(min(num_blocks_in_segment, k - len(available_blocks))):
                    fs_block = (start + i, start + i + required_fs - 1)
                    available_blocks.append(fs_block)
                    if len(available_blocks) >= k:
                        break
            if len(available_blocks) >= k:
                break

        return available_blocks[:k]

    def select_PL(self, pl_list):
        """
        从PL列表中选择一个PL
        选择标准：
        1. 优先选择频谱效率最高的（总容量 / 单link上消耗的FS数量）
        2. 如果频谱效率相同，选择G0中起始频隙最靠前的
        """
        if not pl_list:
            return None

        # 如果只有一个PL，直接返回
        if len(pl_list) == 1:
            return pl_list[0]

        # 计算每个PL的频谱效率和起始频隙
        pl_info = []
        for pl in pl_list:
            transponder_mode = pl["transponder_mode"]
            fs_block = pl["fs_block"]

            # 计算频谱效率 = 总容量 / 单link上消耗的FS数量
            total_capacity = transponder_mode["capacity"]
            fs_consumed = transponder_mode["fs_required"]  # 每个链路消耗的FS数量
            spectral_efficiency = total_capacity / fs_consumed

            # 获取起始频隙
            start_fs = fs_block[0]

            pl_info.append({
                'pl': pl,
                'spectral_efficiency': spectral_efficiency,
                'start_fs': start_fs
            })

        # 按照选择标准排序：
        # 1. 优先频谱效率高的（降序）
        # 2. 然后起始频隙小的（升序）
        sorted_pls = sorted(pl_info,
                            key=lambda x: (-x['spectral_efficiency'], x['start_fs']))

        # 返回最优的PL
        return sorted_pls[0]['pl']


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
        if source not in self.nodes or destination not in self.nodes:
            return None

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