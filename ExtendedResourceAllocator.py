from ExtendCAG import ExtendedCAG
from Tool import *


class ExtendedResourceAllocator:
    def __init__(self, network, K=3, maxnum_PL=5):
        self.network = network
        self.K = K
        self.maxnum_PL = maxnum_PL

    def allocate_demand(self, demand):
        """为需求分配资源"""
        # 构建扩展CAG
        cag = ExtendedCAG(self.network, demand, self.K, self.maxnum_PL)

        # 使用带G0限制的优化SPFA寻找最短路径
        distance, path, edge_sequence, found = optimized_spfa_with_g0_constraint(cag, demand.source, demand.destination)

        if not found:
            return False, "No path found with G0 constraints"

        # 分配资源
        success, message = self.allocate_path_resources(edge_sequence, demand)
        return success, message

    def allocate_path_resources(self, edge_sequence, demand):
        """为找到的路径分配资源"""
        lightpaths_used = []
        otn_switching_nodes = set()

        # 第一个节点是起点
        if edge_sequence:
            otn_switching_nodes.add(edge_sequence[0].get('source',
                                                         getattr(edge_sequence[0].get('lightpath', {}), 'source',
                                                                 None)))

        for i, edge_info in enumerate(edge_sequence):
            # 根据边类型分配资源
            if edge_info["type"] == "EL":
                success = self.allocate_el(edge_info, demand, lightpaths_used)
                # 记录节点
                lp = edge_info["lightpath"]
                otn_switching_nodes.add(lp.source)
                otn_switching_nodes.add(lp.destination)

            elif edge_info["type"] == "EEL":
                success = self.allocate_eel(edge_info, demand, lightpaths_used)
                # 记录节点
                eel_info = edge_info["lightpath"]
                extended_lp = eel_info["extended_lightpath"]
                otn_switching_nodes.add(extended_lp.source)
                otn_switching_nodes.add(extended_lp.destination)

            elif edge_info["type"] == "PL":
                success = self.allocate_pl(edge_info, demand, lightpaths_used)
                # 记录节点
                path_G0 = edge_info["path_G0"]
                otn_switching_nodes.add(path_G0[0])
                otn_switching_nodes.add(path_G0[-1])
            else:
                return False, f"Unknown edge type: {edge_info['type']}"

            if not success:
                return False, f"Failed to allocate resources for edge {i}"

        # 更新OTN交换容量
        for node in otn_switching_nodes:
            if node is not None:
                self.network.update_otn_switching(node, demand.traffic_class.value)

        # 记录分配信息
        demand.path = [edge_sequence[0].get('source',
                                            getattr(edge_sequence[0].get('lightpath', {}), 'source',
                                                    None))] if edge_sequence else []
        for edge_info in edge_sequence:
            if edge_info["type"] == "EL":
                demand.path.append(edge_info["lightpath"].destination)
            elif edge_info["type"] == "EEL":
                demand.path.append(edge_info["lightpath"]["extended_lightpath"].destination)
            elif edge_info["type"] == "PL":
                demand.path.append(edge_info["path_G0"][-1])

        demand.lightpaths_used = lightpaths_used

        return True, "Allocation successful with G0 constraints"

    def allocate_el(self, edge_info, demand, lightpaths_used):
        """分配现有光路资源"""
        lightpath = edge_info["lightpath"]
        if lightpath.add_demand(demand):
            lightpaths_used.append(lightpath)
            return True
        return False

    def allocate_eel(self, edge_info, demand, lightpaths_used):
        """分配可扩展光路资源"""
        eel_info = edge_info["lightpath"]
        original_lp = eel_info["original_lightpath"]
        extended_lp = eel_info["extended_lightpath"]

        # 移除原有光路，创建扩展后的光路
        self.network.remove_lightpath(original_lp)
        new_lp = self.network.create_lightpath(
            extended_lp.source, extended_lp.destination,
            extended_lp.transponder_mode, extended_lp.fs_allocated,
            extended_lp.path_in_G0
        )

        # 迁移原有需求
        for d in original_lp.demands:
            new_lp.add_demand(d)
            if original_lp in d.lightpaths_used:
                index = d.lightpaths_used.index(original_lp)
                d.lightpaths_used[index] = new_lp

        # 添加新需求
        if new_lp.add_demand(demand):
            lightpaths_used.append(new_lp)
            return True
        return False

    def allocate_pl(self, edge_info, demand, lightpaths_used):
        """分配潜在光路资源"""
        transponder_mode = edge_info["transponder_mode"]
        path_G0 = edge_info["path_G0"]
        fs_block = edge_info["fs_block"]

        lightpath = self.network.create_lightpath(
            path_G0[0], path_G0[-1], transponder_mode, fs_block, path_G0
        )

        if lightpath.add_demand(demand):
            lightpaths_used.append(lightpath)
            return True
        return False

# class ExtendedResourceAllocator:
#     def __init__(self, network, K=3, maxnum_PL=5):
#         self.network = network
#         self.K = K
#         self.maxnum_PL = maxnum_PL
#
#     def allocate_demand(self, demand):
#         """为需求分配资源"""
#         # 构建扩展CAG
#         cag = ExtendedCAG(self.network, demand, self.K, self.maxnum_PL)
#
#         # 获取图表示
#         graph = cag.get_graph_representation()
#
#         # 使用优化的SPFA寻找最短路径
#         distance, path, found = optimized_spfa(graph, demand.source, demand.destination)
#
#         if not found:
#             return False, "No path found"
#
#         # 分配资源
#         success, message = self.allocate_path_resources(cag, path, demand)
#         return success, message
#
#     def allocate_path_resources(self, cag, path, demand):
#         """为找到的路径分配资源"""
#         lightpaths_used = []
#         otn_switching_nodes = set()
#
#         for i in range(len(path) - 1):
#             u = path[i]
#             v = path[i + 1]
#
#             # 在CAG中找到对应的边信息
#             edge_info = self.find_best_edge(cag, u, v)
#             if not edge_info:
#                 return False, f"No available edge between {u} and {v}"
#
#             # 根据边类型分配资源
#             if edge_info["type"] == "EL":
#                 success = self.allocate_el(edge_info, demand, lightpaths_used)
#             elif edge_info["type"] == "EEL":
#                 success = self.allocate_eel(edge_info, demand, lightpaths_used)
#             elif edge_info["type"] == "PL":
#                 success = self.allocate_pl(edge_info, demand, lightpaths_used)
#             else:
#                 return False, f"Unknown edge type: {edge_info['type']}"
#
#             if not success:
#                 return False, f"Failed to allocate resources for edge {u}-{v}"
#
#             # 记录OTN交换节点
#             if i == 0:
#                 otn_switching_nodes.add(u)
#             otn_switching_nodes.add(v)
#
#         # 更新OTN交换容量
#         for node in otn_switching_nodes:
#             self.network.update_otn_switching(node, demand.traffic_class.value)
#
#         # 记录分配信息
#         demand.path = path
#         demand.lightpaths_used = lightpaths_used
#
#         return True, "Allocation successful"
#
#     def find_best_edge(self, cag, u, v):
#         """在CAG中找到u到v的最佳边（权重最小）"""
#         if u not in cag.edges:
#             return None
#
#         best_edge = None
#         best_weight = float('inf')
#
#         for target, edge_info in cag.edges[u]:
#             if target == v and edge_info["weight"] < best_weight:
#                 best_edge = edge_info
#                 best_weight = edge_info["weight"]
#
#         return best_edge
#
#     def allocate_el(self, edge_info, demand, lightpaths_used):
#         """分配现有光路资源"""
#         lightpath = edge_info["lightpath"]
#         if lightpath.add_demand(demand):
#             lightpaths_used.append(lightpath)
#             return True
#         return False
#
#     def allocate_eel(self, edge_info, demand, lightpaths_used):
#         """分配可扩展光路资源"""
#         eel_info = edge_info["lightpath"]
#         original_lp = eel_info["original_lightpath"]
#         extended_lp = eel_info["extended_lightpath"]
#
#         # 移除原有光路，创建扩展后的光路
#         self.network.remove_lightpath(original_lp)
#         new_lp = self.network.create_lightpath(
#             extended_lp.source, extended_lp.destination,
#             extended_lp.transponder_mode, extended_lp.fs_allocated,
#             extended_lp.path_in_G0
#         )
#
#         # 迁移原有需求
#         for d in original_lp.demands:
#             new_lp.add_demand(d)
#             if original_lp in d.lightpaths_used:
#                 index = d.lightpaths_used.index(original_lp)
#                 d.lightpaths_used[index] = new_lp
#
#         # 添加新需求
#         if new_lp.add_demand(demand):
#             lightpaths_used.append(new_lp)
#             return True
#         return False
#
#     def allocate_pl(self, edge_info, demand, lightpaths_used):
#         """分配潜在光路资源"""
#         transponder_mode = edge_info["transponder_mode"]
#         path_G0 = edge_info["path_G0"]
#         fs_block = edge_info["fs_block"]
#
#         lightpath = self.network.create_lightpath(
#             path_G0[0], path_G0[-1], transponder_mode, fs_block, path_G0
#         )
#
#         if lightpath.add_demand(demand):
#             lightpaths_used.append(lightpath)
#             return True
#         return False