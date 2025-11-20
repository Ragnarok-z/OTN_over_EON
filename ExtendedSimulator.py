from Simulator import Simulator
from ExtendedResourceAllocator import ExtendedResourceAllocator
from Defragmentation import DefragmentationEngine


class ExtendedSimulator(Simulator):
    def __init__(self, network, traffic_intensity=10, num_demands=1000,
                 random_seed=423, defrag_params={}, output_dir=None,
                 K=3, maxnum_PL=5):
        super().__init__(network, traffic_intensity, num_demands,
                         random_seed, defrag_params, output_dir)
        self.allocator = ExtendedResourceAllocator(network, K, maxnum_PL)

    def process_arrival(self, demand, policy="MinPB", K=3, max_hops=5):
        """处理需求到达事件（使用新的资源分配方案）"""
        success, message = self.allocator.allocate_demand(demand)

        # if demand.id == 100:
        #     print("100", demand.path)

        if not success:
            self.blocked_demands.append(demand)
            if self.defrag_params["en"]:
                # 执行碎片整理
                defrag_engine = DefragmentationEngine(self.network)
                defrag_engine.trigger_defragmentation(demand)
            return

        # 记录成功分配的需求
        self.active_demands[demand.id] = demand

    def process_departure(self, demand):
        """处理需求离开事件（保持不变）"""
        if demand.id not in self.active_demands:
            return

        # 释放资源
        for lightpath in demand.lightpaths_used:
            lightpath.remove_demand(demand)

            if len(lightpath.demands) == 0 and lightpath in self.network.lightpaths:
                self.network.remove_lightpath(lightpath)

        # 更新OTN交换容量
        if demand.path:
            otn_switching_nodes = set(demand.path)  # 包含所有节点
            for node in otn_switching_nodes:
                assert node is not None, (demand.path, demand.id)
                self.network.update_otn_switching(node, -demand.traffic_class.value)

        self.completed_demands.append(demand)
        del self.active_demands[demand.id]