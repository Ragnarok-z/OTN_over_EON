import collections
import math
import heapq
from typing import List, Dict, Set, Tuple, Optional

from Lightpath import Lightpath


class ExtendedCAG:
    def __init__(self, network, demand, K=3, maxnum_PL=5):
        self.network = network
        self.demand = demand
        self.K = K
        self.maxnum_PL = maxnum_PL
        self.nodes = set()
        # 使用双层字典结构：{u: {v: [edge_info1, edge_info2, ...]}}
        self.edges = collections.defaultdict(lambda: collections.defaultdict(list))
        # 预计算最大bitrate/num_fs比值
        self.max_bitrate_per_fs = self.network.get_max_bitrate_per_fs()
        self.build_extended_cag()

    def build_extended_cag(self):
        """构建扩展的CAG，包含所有可能的EL、EEL和有限数量的PL"""
        # 找到K最短路径
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

        # 为每对节点添加所有可能的边
        for u in self.nodes:
            for v in self.nodes:
                if u == v:
                    continue

                # 1. 添加所有EL
                existing_lps = self.network.find_existing_lightpaths(u, v)
                for lp in existing_lps:
                    if lp.can_accommodate(self.demand):
                        edge_info = {
                            'source': u,
                            "type": "EL",
                            "lightpath": lp,
                            "weight": self.calc_lp_score(lp),
                            "g0_links": self.get_g0_links_for_lightpath(lp)
                        }
                        self.edges[u][v].append(edge_info)

                # 如果已经有EL，跳过EEL和PL（根据需求优先级）
                if existing_lps and any(lp.can_accommodate(self.demand) for lp in existing_lps):
                    continue

                # 2. 添加所有EEL
                extendable_lps = self.network.find_extendable_lightpaths(u, v, self.demand)
                for eel_info in extendable_lps:
                    extended_lp = eel_info["extended_lightpath"]
                    edge_info = {
                        'source': u,
                        "type": "EEL",
                        "lightpath": eel_info,
                        "weight": self.calc_lp_score(extended_lp),
                        "g0_links": self.get_g0_links_for_lightpath(extended_lp)
                    }
                    self.edges[u][v].append(edge_info)

                # 如果已经有EEL，跳过PL
                if extendable_lps:
                    continue

                # 3. 添加最多maxnum_PL个PL
                pl_count = 0
                # 找到K条最短路径用于PL
                pl_paths = self.network.find_k_shortest_paths(u, v, self.maxnum_PL)

                for path_G0 in pl_paths:
                    if pl_count >= self.maxnum_PL:
                        break

                    # 检查是否可以创建光路
                    can_create, mode, _, fs_block = self.network.can_create_lightpath_for_path(path_G0, self.demand)
                    if can_create:
                        edge_info = {
                            "type": "PL",
                            'source': u,
                            "lightpath": Lightpath(u,v,mode,fs_block,path_G0),
                            "transponder_mode": mode,
                            "path_G0": path_G0,
                            "fs_block": fs_block,
                            "weight": self.calc_pl_score(mode, path_G0),
                            "g0_links": self.get_g0_links_for_path(path_G0)
                        }
                        self.edges[u][v].append(edge_info)
                        pl_count += 1

    def get_g0_links_for_lightpath(self, lightpath):
        """获取光路占用的G0层链路"""
        g0_links = set()
        path_G0 = lightpath.path_in_G0
        for i in range(len(path_G0) - 1):
            u, v = path_G0[i], path_G0[i + 1]
            # 使用排序后的元组表示无向边
            g0_links.add(tuple(sorted((u, v))))
        return g0_links

    def get_g0_links_for_path(self, path_G0):
        """获取路径占用的G0层链路"""
        g0_links = set()
        for i in range(len(path_G0) - 1):
            u, v = path_G0[i], path_G0[i + 1]
            g0_links.add(tuple(sorted((u, v))))
        return g0_links

    def calc_lp_score(self, lightpath) -> float:
        """
        计算光路的完整分数: η_lp = η_fsclp * hop_lp * num_fs

        其中:
        η_fsclp = (bitrate/num_fs)/max_bitrate_per_fs * (hop_Ls/hop_lp) *
                  (Σ_{e∈Ls} C_B(e)) / (Σ_{e∈L} C_B(e)) * (1 - Δf)
        """
        try:
            # 获取光路基本信息
            bitrate = lightpath.transponder_mode["bitrate"]
            num_fs = lightpath.transponder_mode["fs_required"]
            path_G0 = lightpath.path_in_G0
            hop_lp = len(path_G0) - 1

            if hop_lp == 0:
                return float('inf')

            # 1. 计算频谱效率部分: (bitrate/num_fs)/max_bitrate_per_fs
            spectral_efficiency = (bitrate / num_fs) / self.max_bitrate_per_fs

            # 2. 计算最短路径跳数比例: hop_Ls / hop_lp
            shortest_path = self.network.dijkstra(lightpath.source, lightpath.destination)
            if not shortest_path:
                hop_Ls = hop_lp  # 如果没有最短路径，使用当前跳数
            else:
                hop_Ls = len(shortest_path) - 1

            hop_ratio = hop_Ls / hop_lp

            # 3. 计算介数比例: Σ_{e∈Ls} C_B(e) / Σ_{e∈L} C_B(e)
            # 计算最短路径的介数和
            shortest_path_betweenness_sum = 0.0
            if shortest_path:
                for i in range(len(shortest_path) - 1):
                    u, v = shortest_path[i], shortest_path[i + 1]
                    shortest_path_betweenness_sum += self.network.get_edge_betweenness(u, v)

            # 计算当前路径的介数和
            current_path_betweenness_sum = 0.0
            for i in range(len(path_G0) - 1):
                u, v = path_G0[i], path_G0[i + 1]
                current_path_betweenness_sum += self.network.get_edge_betweenness(u, v)

            if current_path_betweenness_sum == 0:
                betweenness_ratio = 1.0
            else:
                betweenness_ratio = shortest_path_betweenness_sum / current_path_betweenness_sum

            # 4. 计算OTN碎片变化 Δf (简化计算)
            # delta_f = self.calculate_otn_fragmentation(lightpath)
            delta_f=0

            # 5. 计算 η_fsclp
            eta_fsclp = (spectral_efficiency * hop_ratio *
                         betweenness_ratio * (1 - delta_f))

            # 6. 计算最终分数 η_lp
            # eta_lp = eta_fsclp * hop_lp * num_fs
            eta_lp = eta_fsclp * hop_lp * self.demand.traffic_class.value/10

            # 确保分数为正数
            return max(eta_lp, 0.001)

        except Exception as e:
            print(f"Error calculating LP score: {e}")
            return 1.0  # 出错时返回默认值

    def calculate_otn_fragmentation(self, lightpath) -> float:
        """
        计算OTN层碎片变化Δf

        简化实现: 基于平行光路的剩余容量计算碎片
        """
        source, destination = lightpath.source, lightpath.destination

        # 找到源-目的地之间的所有平行光路
        parallel_lightpaths = []
        for lp in self.network.lightpaths:
            if lp.source == source and lp.destination == destination:
                parallel_lightpaths.append(lp)

        if not parallel_lightpaths:
            return 0.0  # 没有平行光路，无碎片

        # 计算剩余容量列表
        remaining_capacities = [lp.remaining_capacity() for lp in parallel_lightpaths]
        total_capacity = sum(remaining_capacities)

        # 计算当前阻塞率（简化模型）
        current_blocking = self.calculate_blocking_probability(remaining_capacities)

        # 计算理想阻塞率（所有容量集中在一个光路）
        ideal_blocking = self.calculate_blocking_probability([total_capacity])

        # Δf = 理想阻塞率 - 当前阻塞率
        delta_f = ideal_blocking - current_blocking

        return max(0.0, min(1.0, delta_f))

    def calculate_blocking_probability(self, capacities, traffic_intensity=1.0):
        """
        计算阻塞概率（简化版Erlang B公式）

        Args:
            capacities: 容量列表
            traffic_intensity: 流量强度

        Returns:
            阻塞概率
        """
        if not capacities:
            return 1.0

        # 简化实现：使用修正的Erlang B公式
        total_capacity = sum(capacities)
        if total_capacity == 0:
            return 1.0

        # 简化的阻塞概率计算
        blocking_prob = traffic_intensity / (traffic_intensity + total_capacity)
        return blocking_prob

    def calc_pl_score(self, transponder_mode, path_G0) -> float:
        """
        计算潜在光路的分数

        对于PL，我们创建一个虚拟的lightpath对象来计算分数
        """
        try:
            # 创建虚拟lightpath对象
            class VirtualLightpath:
                def __init__(self, source, destination, transponder_mode, path_G0):
                    self.source = source
                    self.destination = destination
                    self.transponder_mode = transponder_mode
                    self.path_in_G0 = path_G0
                    self.used_capacity = 0
                    self.demands = []

                def remaining_capacity(self):
                    return self.transponder_mode["capacity"] - self.used_capacity

            if len(path_G0) < 2:
                return float('inf')

            virtual_lp = VirtualLightpath(path_G0[0], path_G0[-1], transponder_mode, path_G0)
            return self.calc_lp_score(virtual_lp)

        except Exception as e:
            print(f"Error calculating PL score: {e}")
            return 1.0

    def get_all_edges_between(self, u, v):
        """获取u到v之间的所有边"""
        return self.edges[u][v]

    def get_graph_representation(self):
        """获取图表示，用于SPFA算法（简化版本，只包含最小权重）"""
        graph = {}
        for u in self.nodes:
            graph[u] = {}
            for v in self.edges[u]:
                edges_list = self.edges[u][v]
                if edges_list:
                    # 选择权重最小的边
                    best_edge = min(edges_list, key=lambda x: x["weight"])
                    graph[u][v] = best_edge["weight"]
        return graph