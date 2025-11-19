import collections
import math
import heapq
from typing import List, Dict, Set, Tuple, Optional


class ExtendedCAG:
    def __init__(self, network, demand, K=3, maxnum_PL=5):
        self.network = network
        self.demand = demand
        self.K = K
        self.maxnum_PL = maxnum_PL
        self.nodes = set()
        # 使用双层字典结构：{u: {v: [edge_info1, edge_info2, ...]}}
        self.edges = collections.defaultdict(lambda: collections.defaultdict(list))
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
        """计算光路的分数（简化为固定值1）"""
        return 1.0

    def calc_pl_score(self, transponder_mode, path_G0) -> float:
        """计算潜在光路的分数（简化为固定值1）"""
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


# class ExtendedCAG:
#     def __init__(self, network, demand, K=3, maxnum_PL=5):
#         self.network = network
#         self.demand = demand
#         self.K = K
#         self.maxnum_PL = maxnum_PL
#         self.nodes = set()
#         self.edges = collections.defaultdict(list)  # 改为存储多个边 {u: {v: [edge_info1, edge_info2, ...]}}
#         self.build_extended_cag()
#
#     def build_extended_cag(self):
#         """构建扩展的CAG，包含所有可能的EL、EEL和有限数量的PL"""
#         # 找到K最短路径
#         k_shortest_paths = self.network.find_k_shortest_paths(
#             self.demand.source, self.demand.destination, self.K)
#
#         # Collect all nodes from these paths that have OTN switching capability
#         # self.nodes.add(self.demand.source)
#         # self.nodes.add(self.demand.destination)
#
#         for path in k_shortest_paths:
#             for node in path:
#                 if (self.network.otn_switches[node]["used_capacity"] + self.demand.traffic_class.value
#                         <= self.network.otn_switches[node]["total_capacity"]):
#                     self.nodes.add(node)
#
#         # 为每对节点添加所有可能的边
#         for u in self.nodes:
#             for v in self.nodes:
#                 if u == v:
#                     continue
#
#                 # 1. 添加所有EL
#                 existing_lps = self.network.find_existing_lightpaths(u, v)
#                 for lp in existing_lps:
#                     if lp.can_accommodate(self.demand):
#                         edge_info = {
#                             "type": "EL",
#                             "lightpath": lp,
#                             "weight": self.calc_lp_score(lp)
#                         }
#                         self.edges[u].append((v, edge_info))
#
#                 # 如果已经有EL，跳过EEL和PL（根据需求优先级）
#                 if existing_lps and any(lp.can_accommodate(self.demand) for lp in existing_lps):
#                     continue
#
#                 # 2. 添加所有EEL
#                 extendable_lps = self.network.find_extendable_lightpaths(u, v, self.demand)
#                 for eel_info in extendable_lps:
#                     edge_info = {
#                         "type": "EEL",
#                         "lightpath": eel_info,
#                         "weight": self.calc_lp_score(eel_info["extended_lightpath"])
#                     }
#                     self.edges[u].append((v, edge_info))
#
#                 # 如果已经有EEL，跳过PL
#                 if extendable_lps:
#                     continue
#
#                 # 3. 添加最多maxnum_PL个PL
#                 pl_count = 0
#                 # 找到K条最短路径用于PL
#                 pl_paths = self.network.find_k_shortest_paths(u, v, self.maxnum_PL)
#
#                 for path_G0 in pl_paths:
#                     if pl_count >= self.maxnum_PL:
#                         break
#
#                     # 检查是否可以创建光路
#                     can_create, mode, _, fs_block = self.network.can_create_lightpath_for_path(path_G0, self.demand)
#                     if can_create:
#                         edge_info = {
#                             "type": "PL",
#                             "transponder_mode": mode,
#                             "path_G0": path_G0,
#                             "fs_block": fs_block,
#                             "weight": self.calc_pl_score(mode, path_G0)
#                         }
#                         self.edges[u].append((v, edge_info))
#                         pl_count += 1
#
#     def calc_lp_score(self, lightpath) -> float:
#         """计算光路的分数（简化为固定值1）"""
#         return 1.0
#
#     def calc_pl_score(self, transponder_mode, path_G0) -> float:
#         """计算潜在光路的分数（简化为固定值1）"""
#         return 1.0
#
#     def get_graph_representation(self):
#         """获取图表示，用于SPFA算法"""
#         graph = {}
#         for u in self.nodes:
#             graph[u] = {}
#             for v, edge_info in self.edges[u]:
#                 # 对于每个目标节点，选择权重最小的边（或者可以根据其他策略选择）
#                 if v not in graph[u] or edge_info["weight"] < graph[u][v]:
#                     graph[u][v] = edge_info["weight"]
#         return graph