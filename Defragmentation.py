from collections import defaultdict
import heapq
from typing import List, Set, Tuple, Dict
from Lightpath import Lightpath
from Demand import Demand
from Network import Network
# from Simulator import Simulator

class DefragmentationEngine:
    def __init__(self, network: Network):
        self.network = network

    def trigger_defragmentation(self, blocked_demand: Demand) -> bool:
        """
        触发碎片整理流程
        Args:
            blocked_demand: 被阻塞的业务需求
        Returns:
            bool: 整理是否成功（是否释放了资源）
        """
        # 获取G0层a到b的最短路径
        shortest_path = self.network.dijkstra(blocked_demand.source, blocked_demand.destination)
        if not shortest_path:
            return False

        print(f"触发碎片整理，阻塞需求: {blocked_demand.source}->{blocked_demand.destination}")
        print(f"最短路径: {shortest_path}")

        # 获取所有需要整理的平行光路
        parallel_lightpaths = self.get_parallel_lightpaths(shortest_path)

        # 执行碎片整理
        success = self.perform_defragmentation(parallel_lightpaths)

        return success

    def get_parallel_lightpaths(self, path: List[int]) -> Dict[Tuple[int, int], List[Lightpath]]:
        """
        获取最短路径上所有节点对之间的平行光路
        Args:
            path: G0层最短路径节点列表
        Returns:
            字典: {(source, destination): [lightpath1, lightpath2, ...]}
        """
        parallel_lightpaths = defaultdict(list)

        # 遍历路径上所有节点对
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                source = path[i]
                destination = path[j]

                # 查找该节点对之间的所有光路
                lightpaths = [lp for lp in self.network.lightpaths
                              if lp.source == source and lp.destination == destination]

                if lightpaths:
                    parallel_lightpaths[(source, destination)] = lightpaths

        print(f"找到 {len(parallel_lightpaths)} 组平行光路需要整理")
        return parallel_lightpaths

    def perform_defragmentation(self, parallel_lightpaths: Dict[Tuple[int, int], List[Lightpath]]) -> bool:
        """
        执行碎片整理
        Args:
            parallel_lightpaths: 需要整理的平行光路字典
        Returns:
            bool: 是否成功释放了资源
        """
        removed_count = 0
        moved_demands = 0

        for (source, destination), lightpaths in parallel_lightpaths.items():
            # 按FS分配位置排序光路（编号低的在前）
            sorted_lightpaths = sorted(lightpaths, key=lambda lp: lp.fs_allocated[0])

            # 统计总需求和总容量
            total_demand = sum(lp.used_capacity for lp in sorted_lightpaths)
            total_capacity = sum(lp.capacity for lp in sorted_lightpaths)

            print(f"整理 {source}->{destination}: {len(lightpaths)}条光路, "
                  f"总需求: {total_demand}, 总容量: {total_capacity}")

            # 如果总需求超过总容量，无法通过整理解决
            if total_demand > total_capacity:
                print(f"  容量不足，无法通过整理解决")
                continue

            # 将需求尽量迁移到编号低的光路中
            target_lightpaths = []
            source_lightpaths = []

            # 分离目标光路和源光路
            for i, lp in enumerate(sorted_lightpaths):
                if i < len(sorted_lightpaths) - 1:  # 前n-1条作为目标
                    target_lightpaths.append(lp)
                else:  # 最后一条作为源
                    source_lightpaths.append(lp)

            # 迁移需求
            for source_lp in source_lightpaths:
                if source_lp.used_capacity == 0:
                    continue

                # 复制需求列表（避免在迭代中修改）
                demands_to_move = source_lp.demands.copy()

                for demand in demands_to_move:
                    # 尝试将需求迁移到目标光路
                    moved = False
                    for target_lp in target_lightpaths:
                        if target_lp.can_accommodate(demand):
                            # 执行迁移
                            if self.move_demand(demand, source_lp, target_lp):
                                moved = True
                                moved_demands += 1
                                break

                    if not moved:
                        print(f"  需求 {demand.id} 无法迁移")

            # 检查并移除空光路
            for lp in sorted_lightpaths:
                if len(lp.demands) == 0 and lp in self.network.lightpaths:
                    self.network.remove_lightpath(lp)
                    removed_count += 1
                    print(f"  移除空光路: {lp.source}->{lp.destination}")

        print(f"碎片整理完成: 迁移了 {moved_demands} 个需求, 移除了 {removed_count} 条空光路")
        return removed_count > 0

    def move_demand(self, demand: Demand, source_lp: Lightpath, target_lp: Lightpath) -> bool:
        """
        将需求从一个光路迁移到另一个光路
        Args:
            demand: 要迁移的需求
            source_lp: 源光路
            target_lp: 目标光路
        Returns:
            bool: 迁移是否成功
        """
        # 从源光路移除需求
        if not source_lp.remove_demand(demand):
            return False

        # 添加到目标光路
        if not target_lp.add_demand(demand):
            # 如果添加失败，回滚到源光路
            source_lp.add_demand(demand)
            return False

        # 更新需求的光路使用记录
        if source_lp in demand.lightpaths_used:
            index = demand.lightpaths_used.index(source_lp)
            demand.lightpaths_used[index] = target_lp

        print(f"  迁移需求 {demand.id} ({demand.traffic_class.value}G): "
              f"{source_lp.source}->{source_lp.destination} -> "
              f"{target_lp.source}->{target_lp.destination}")

        return True


# 在Simulator类中添加碎片整理功能
def enhanced_process_arrival(self, demand, policy, K, max_hops):
    """
    增强的到达处理，包含碎片整理
    """
    # 首先尝试正常处理
    original_process_arrival = getattr(self, '_original_process_arrival', None)
    if original_process_arrival is None:
        # 保存原始方法引用
        self._original_process_arrival = self.process_arrival
        original_process_arrival = self.process_arrival

    # 尝试正常处理
    original_process_arrival(demand, policy, K, max_hops)

    # 如果需求被阻塞，触发碎片整理
    if demand in self.blocked_demands:
        print(f"需求 {demand.id} 被阻塞，触发碎片整理...")

        # 初始化碎片整理引擎
        defrag_engine = DefragmentationEngine(self.network)

        # 执行碎片整理
        if defrag_engine.trigger_defragmentation(demand):
            # 整理后重试需求分配
            print("碎片整理完成，重试需求分配...")

            # 从阻塞列表中移除（暂时）
            self.blocked_demands.remove(demand)

            # 重试处理
            original_process_arrival(demand, policy, K, max_hops)

            # 如果仍然阻塞，加回阻塞列表
            if demand not in self.active_demands and demand not in self.completed_demands:
                self.blocked_demands.append(demand)
                print(f"需求 {demand.id} 仍然阻塞")


# 在Simulator类中替换原始方法
def enable_defragmentation(simulator_instance):
    """
    启用碎片整理功能
    """
    simulator_instance.process_arrival = lambda demand, policy, K, max_hops: \
        enhanced_process_arrival(simulator_instance, demand, policy, K, max_hops)


# # 使用示例
# if __name__ == "__main__":
#     # 在模拟器中使用
#     network = Network('topology/nsfnet.txt')
#     simulator = Simulator(network, traffic_intensity=300, num_demands=1000)
#
#     # 启用碎片整理
#     enable_defragmentation(simulator)
#
#     # 运行模拟
#     simulator.run(policy="MinPB")