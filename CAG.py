import heapq
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from params import *

class CAG:
    def __init__(self, network, demand, K=3, include_OTN_frag=False, calc_E_k=5, E_loaded=None):
        self.network = network
        self.demand = demand
        self.K = K
        self.include_OTN_frag = include_OTN_frag  # 添加新参数
        # print('init', self.include_OTN_frag)
        self.nodes = set()
        self.edges = defaultdict(dict)  # {u: {v: edge_info}}
        self.calc_E_k = calc_E_k
        self.E_loaded = E_loaded
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

        el_info = []
        if self.include_OTN_frag=='OEFM':
            select = [i for i in range(len(EL_lst))]
            delta_f_OTN = self.calculate_delta_f_OTN(EL_lst,select, self.calc_E_k)
            for i, el in enumerate(EL_lst):
                el_info.append({
                    id: i if i>0 else '000000000000',
                    'el': el,
                    'delta_f_OTN': delta_f_OTN[i]
                })
            sorted_els = sorted(el_info, key=lambda x: (x['delta_f_OTN']))
            # print('in OEFM')
            return sorted_els[0]['el']


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
        return sorted_els[0]['el']

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

        eel_info = []
        eel_lst = [eel['extended_lightpath'] for eel in extendable_lps]
        if self.include_OTN_frag=='OEFM':
            select = [i for i in range(len(eel_lst))]
            delta_f_OTN = self.calculate_delta_f_OTN(eel_lst, select, self.calc_E_k)
            for i, eel in enumerate(extendable_lps):
                eel_info.append({
                    'id': i if i > 0 else '000000000000',
                    'eel': eel,
                    'delta_f_OTN': delta_f_OTN[i]
                })
            sorted_eels = sorted(eel_info, key=lambda x: (x['delta_f_OTN']))
            return sorted_eels[0]['eel']

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

    ##################
    ##  OEFM的相关计算
    ##################
    # 计算E
    def calc_E(self, c, n):
        return self.E_loaded[c//10][n]

    # 计算OEFM
    def calc_OEFM(self, k, cap, sumcap, n, posx, sumx):
        sum_E = 0
        for i in range(k):
            sum_E += self.calc_E(cap[i], n) / (1 + posx[i] / sumx)

        calc_E_sumcap = self.calc_E(sumcap, n)
        if calc_E_sumcap == 0:
            return 0
        return 1 - sum_E / k / calc_E_sumcap

    # 计算delta_f_OTN
    def calculate_delta_f_OTN(self, lps, select, n):

        k = len(lps)
        cap = [lp.remaining_capacity() for lp in lps]
        sumcap = sum(cap)
        posx = [(lp.fs_allocated[0]+lp.fs_allocated[1])/2 for lp in lps]
        for i in range(len(posx)):
            posx[i] = posx[i]+posx[i-1]
        sumx = posx[-1]
        f_OTN_base = self.calc_OEFM(k, cap, sumcap, n, posx, sumx)

        delta_f_OTN = []
        required_cap = self.demand.traffic_class.value
        sumcap -= required_cap
        for i in select:
            cap[i] -= required_cap
            delta_f_OTN.append(self.calc_OEFM(k, cap, sumcap, n, posx, sumx) - f_OTN_base)
            cap[i] += required_cap
        return delta_f_OTN


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

    def calculate_delta_f_EON(self,path_G0, fs_block):
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
            if coeffs["cf_EON"]:
                delta_f_EON = self.calculate_delta_f_EON(path_G0, fs_block)
            else:
                delta_f_EON=0

            if coeffs["cf_OTN"]:
                delta_f_OTN = self.calculate_delta_f_OTN()
            else:
                delta_f_OTN = 0

            u_value = mode["bitrate"]

            weight = (coeffs["c0"] + 1 * (
                    coeffs["c0_new"] +
                    coeffs["c_new"] * h +
                    coeffs["cf_EON"] * delta_f_EON +
                    coeffs["cf_OTN"] * delta_f_OTN +
                    coeffs["c_prime"] * h ** 2 +
                    coeffs["cu"] * 10 ** (-u_value)
            ))
            return weight

    def find_shortest_path(self, policy, max_hops=5, overlap_num=2, sp_algo='base'):
        if sp_algo=='base':
            return self.find_shortest_path_base(policy=policy, max_hops=max_hops)
        elif sp_algo=="LOC-SP-algo":
            return self.find_shortest_path_LOC_SP(policy=policy, max_hops=max_hops, overlap_num=overlap_num)



    def find_shortest_path_base(self, policy, max_hops=5):
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

    def find_shortest_path_LOC_SP(self, policy, max_hops=5, overlap_num=2):
        """
        带有限重叠约束的改进型Dijkstra算法
        输入：
            policy: 流量工程策略（与原函数一致）
            max_hops: 最大跳数约束（与原函数一致）
            overlap_num: 最大允许重叠数（总使用G0链路数 - 唯一G0链路数 ≤ 此值）
        输出：最优路径列表（None表示无可行路径），与原函数输入输出完全一致
        约束：
            1. 硬约束：EEL/PL的G0链路不可与已用EEL/PL链路重复；
            2. 软约束：总重叠数 = 总使用G0链路数 - 唯一G0链路数 ≤ overlap_num；
        """
        source = self.demand.source
        destination = self.demand.destination

        # 边界判断：源/目的不在CAG节点集中
        if source not in self.nodes or destination not in self.nodes:
            return None

        # -------------------------- 状态定义 --------------------------
        # 状态键：(当前节点, 已用EEL/PL链路集(frozenset), 总重叠数, 已跳数)
        # 状态值：到达该状态的最小代价
        state_cost = defaultdict(lambda: float('inf'))
        # 前驱记录：(状态键) → (前驱节点, 链路使用统计字典, 路径列表)
        # 链路使用统计字典：{G0链路元组: 使用次数}
        state_prev = dict()

        # 初始化：源节点，无已用EEL/PL链路，无链路使用，重叠数0，跳数0，代价0
        init_used_eel_pl = frozenset()  # 已用EEL/PL链路（硬约束）
        init_link_count = defaultdict(int)  # 所有链路使用统计（软约束）
        init_overlap = 0  # 初始重叠数=0
        init_hops = 0
        init_state = (source, init_used_eel_pl, init_overlap, init_hops)
        state_cost[init_state] = 0
        state_prev[init_state] = (None, init_link_count, [source])

        # 优先级队列：(总代价, 当前节点, 已用EEL/PL链路集, 总重叠数, 已跳数)
        pq = []
        heapq.heappush(pq, (0, source, init_used_eel_pl, init_overlap, init_hops))

        # 最优路径记录
        best_path = None
        min_total_cost = float('inf')

        # -------------------------- 核心遍历 --------------------------
        while pq:
            # 弹出当前最小代价状态
            curr_cost, curr_node, curr_used_eel_pl, curr_overlap, curr_hops = heapq.heappop(pq)

            # 剪枝1：当前代价已大于已知最优解
            if curr_cost > min_total_cost:
                continue

            # 剪枝2：跳数超限（无法继续扩展）
            if curr_hops >= max_hops:
                # 若到达目的节点，更新最优解
                if curr_node == destination and curr_cost < min_total_cost:
                    min_total_cost = curr_cost
                    best_path = state_prev[(curr_node, curr_used_eel_pl, curr_overlap, curr_hops)][2]
                continue

            # 到达目的节点，更新最优解
            if curr_node == destination:
                min_total_cost = curr_cost
                best_path = state_prev[(curr_node, curr_used_eel_pl, curr_overlap, curr_hops)][2]
                continue

            # 遍历当前节点的所有出边
            for neighbor in self.edges.get(curr_node, {}):
                edge_info = self.edges[curr_node][neighbor]
                # 1. 计算当前边的权重（与原逻辑一致）
                edge_weight = self.calculate_edge_weight(curr_node, neighbor, policy)
                new_cost = curr_cost + edge_weight
                new_hops = curr_hops + 1

                # 2. 提取当前边的G0链路（统一格式：tuple(sorted((u,v))) 保证无向）
                if edge_info["type"] == "EL":
                    # EL链路：提取其G0路径
                    lightpath = edge_info["lightpath"]
                    edge_g0_links = [tuple(sorted((lightpath.path_in_G0[i], lightpath.path_in_G0[i + 1])))
                                     for i in range(len(lightpath.path_in_G0) - 1)]
                    is_eel_pl = False  # EL不参与硬约束
                elif edge_info["type"] == "EEL":
                    # EEL链路：提取扩展后的G0路径
                    extended_lp = edge_info["lightpath"]["extended_lightpath"]
                    edge_g0_links = [tuple(sorted((extended_lp.path_in_G0[i], extended_lp.path_in_G0[i + 1])))
                                     for i in range(len(extended_lp.path_in_G0) - 1)]
                    is_eel_pl = True  # EEL参与硬约束
                else:  # PL
                    # PL链路：提取新建的G0路径
                    path_G0 = edge_info["path_G0"]
                    edge_g0_links = [tuple(sorted((path_G0[i], path_G0[i + 1])))
                                     for i in range(len(path_G0) - 1)]
                    is_eel_pl = True  # PL参与硬约束

                # 3. 硬约束检查：EEL/PL链路不可与已用EEL/PL链路重复
                if is_eel_pl:
                    # 检查当前EEL/PL的G0链路是否与已用EEL/PL链路重叠
                    if set(edge_g0_links) & set(curr_used_eel_pl):
                        continue  # 硬约束冲突，跳过
                    # 新的已用EEL/PL链路集
                    new_used_eel_pl = frozenset(set(curr_used_eel_pl) | set(edge_g0_links))
                else:
                    new_used_eel_pl = curr_used_eel_pl  # EL不改变已用EEL/PL链路集

                # 4. 软约束计算：更新链路使用统计和总重叠数
                # 复制前驱的链路使用统计（避免修改原字典）
                prev_link_count = state_prev[(curr_node, curr_used_eel_pl, curr_overlap, curr_hops)][1]
                new_link_count = defaultdict(int, prev_link_count)

                # 遍历当前边的所有G0链路，更新使用次数
                for link in edge_g0_links:
                    new_link_count[link] += 1

                # 计算新的总重叠数：总使用数 - 唯一数
                total_used = sum(new_link_count.values())
                unique_used = len(new_link_count)
                new_overlap = total_used - unique_used

                # 软约束检查：重叠数超限则跳过
                if new_overlap > overlap_num:
                    continue

                # 5. 状态更新与剪枝
                new_state = (neighbor, new_used_eel_pl, new_overlap, new_hops)
                # 若新状态代价更高，跳过
                if new_cost >= state_cost[new_state]:
                    continue

                # 更新状态代价和前驱
                state_cost[new_state] = new_cost
                # 拼接新路径
                prev_path = state_prev[(curr_node, curr_used_eel_pl, curr_overlap, curr_hops)][2]
                new_path = prev_path + [neighbor]
                state_prev[new_state] = (curr_node, new_link_count, new_path)

                # 加入优先级队列
                heapq.heappush(pq, (new_cost, neighbor, new_used_eel_pl, new_overlap, new_hops))

        # 返回最优路径（无可行路径则返回None）
        return best_path