import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from collections import defaultdict, deque
import random
from tqdm import tqdm
import math
from scipy.stats import poisson, expon
from itertools import combinations


class OpticalNetworkSimulator:
    def __init__(self, topology_file):
        # 初始化网络参数
        self.params = {
            'fs_bandwidth': 6.25,  # GHz
            'total_fs': 768,
            'c_band_start': 191.3,  # THz
            'c_band_end': 196.1,  # THz
            'otn_switch_capacity': 24000,  # Gb/s
            'ethernet_rates': [10, 100, 400],  # GE
            'transponder_modes': self._init_transponder_modes(),
            'k_shortest_paths': 3,
            'max_hops': 5,
            'max_otn_switching': float('inf')
        }

        # 初始化网络拓扑
        self.topology = self._load_topology(topology_file)
        self._init_network_layers()

        # 性能指标
        self.metrics = {
            'blocking_ratio': [],
            'spectrum_usage': [],
            'avg_hops': [],
            'otn_switching': [],
            'multi_demand_lightpaths': [],
            'lightpath_utilization': [],
            'throughput': []
        }

        # 频谱分配状态
        self.spectrum_allocations = {edge: set() for edge in self.topology['physical'].edges()}

    def _init_transponder_modes(self):
        # 严格遵循论文表V中的收发器模式
        return [
            {'capacity': 200, 'modem_rate': 200, 'baudrate': 95, 'fs': 19, 'reach': 125},
            {'capacity': 300, 'modem_rate': 300, 'baudrate': 95, 'fs': 19, 'reach': 88},
            {'capacity': 400, 'modem_rate': 400, 'baudrate': 95, 'fs': 19, 'reach': 54},
            {'capacity': 500, 'modem_rate': 500, 'baudrate': 95, 'fs': 19, 'reach': 35},
            {'capacity': 600, 'modem_rate': 600, 'baudrate': 95, 'fs': 19, 'reach': 18},
            {'capacity': 700, 'modem_rate': 700, 'baudrate': 95, 'fs': 19, 'reach': 9},
            {'capacity': 800, 'modem_rate': 800, 'baudrate': 95, 'fs': 19, 'reach': 4},
            {'capacity': 100, 'modem_rate': 100, 'baudrate': 56, 'fs': 12, 'reach': 130},
            {'capacity': 200, 'modem_rate': 200, 'baudrate': 56, 'fs': 12, 'reach': 61},
            {'capacity': 300, 'modem_rate': 300, 'baudrate': 56, 'fs': 12, 'reach': 34},
            {'capacity': 400, 'modem_rate': 400, 'baudrate': 56, 'fs': 12, 'reach': 10},
            {'capacity': 100, 'modem_rate': 100, 'baudrate': 35, 'fs': 8, 'reach': 75},
            {'capacity': 200, 'modem_rate': 200, 'baudrate': 35, 'fs': 8, 'reach': 16}
        ]

    def _load_topology(self, filename):
        """从文件加载拓扑结构"""
        topology = {'physical': nx.Graph(), 'lightpath': nx.Graph(), 'tunnel': nx.Graph()}

        with open(filename, 'r') as f:
            n, m = map(int, f.readline().split())
            for _ in range(m):
                s, t, d = map(int, f.readline().split())
                topology['physical'].add_edge(s, t, length=d)

        # 初始化其他层
        topology['lightpath'].add_nodes_from(topology['physical'].nodes())
        topology['tunnel'].add_nodes_from(topology['physical'].nodes())

        return topology

    def _init_network_layers(self):
        """初始化网络各层状态"""
        # G1层状态
        for u, v in self.topology['lightpath'].edges():
            self.topology['lightpath'].edges[u, v]['capacity'] = 0
            self.topology['lightpath'].edges[u, v]['fs'] = 0
            self.topology['lightpath'].edges[u, v]['demands'] = []
            self.topology['lightpath'].edges[u, v]['spectrum'] = set()

        # OTN交换容量
        self.otn_switch_usage = {node: 0 for node in self.topology['physical'].nodes()}

    def _generate_demand(self, current_time, arrival_rate):
        """生成业务需求"""
        rate = random.choice(self.params['ethernet_rates'])
        nodes = list(self.topology['physical'].nodes())
        src, dst = random.sample(nodes, 2)
        duration = expon.rvs(scale=1 / arrival_rate)
        return {
            'src': src,
            'dst': dst,
            'rate': rate,
            'arrival_time': current_time,
            'duration': duration,
            'end_time': current_time + duration,
            'status': 'pending'
        }

    def _build_cag(self, demand):
        """构建折叠辅助图(CAG)"""
        cag = nx.Graph()
        src, dst = demand['src'], demand['dst']
        k = self.params['k_shortest_paths']

        try:
            # 使用Yen算法找到K最短路径
            paths = list(nx.shortest_simple_paths(
                self.topology['physical'], src, dst, weight='length'))[:k]
        except nx.NetworkXNoPath:
            return None

        # 为每条路径构建CAG
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]

                # 检查现有光路径(EL)
                if self.topology['lightpath'].has_edge(u, v):
                    lp = self.topology['lightpath'].edges[u, v]
                    if lp['capacity'] >= demand['rate']:
                        cag.add_edge(u, v,
                                     type='EL',
                                     capacity=lp['capacity'],
                                     used_capacity=sum(d['rate'] for d in lp['demands']),
                                     fs=lp['fs'],
                                     spectrum=lp['spectrum'],
                                     cost=self._calculate_edge_cost('EL', u, v, demand))

                # 检查可扩展光路径(EEL)
                if self.topology['lightpath'].has_edge(u, v):
                    lp = self.topology['lightpath'].edges[u, v]
                    required_additional = demand['rate'] - (lp['capacity'] - sum(d['rate'] for d in lp['demands']))
                    if required_additional > 0:
                        # 检查是否可以扩展
                        if self._can_extend_lightpath(u, v, required_additional):
                            cag.add_edge(u, v,
                                         type='EEL',
                                         capacity=lp['capacity'] + required_additional,
                                         additional=required_additional,
                                         fs=lp['fs'],
                                         spectrum=lp['spectrum'],
                                         cost=self._calculate_edge_cost('EEL', u, v, demand))

                # 检查潜在光路径(PL)
                if not self.topology['lightpath'].has_edge(u, v):
                    distance = nx.shortest_path_length(self.topology['physical'], u, v, weight='length')
                    best_mode = self._select_transponder_mode(distance, demand['rate'])
                    if best_mode:
                        # 检查频谱可用性
                        if self._check_spectrum_availability(u, v, best_mode['fs']):
                            cag.add_edge(u, v,
                                         type='PL',
                                         capacity=best_mode['capacity'],
                                         fs=best_mode['fs'],
                                         mode=best_mode,
                                         cost=self._calculate_edge_cost('PL', u, v, demand))

        return cag if cag.number_of_edges() > 0 else None

    def _can_extend_lightpath(self, u, v, additional_capacity):
        """检查是否可以扩展光路径容量"""
        # 在实际实现中需要考虑OTN复用限制等
        return True

    def _select_transponder_mode(self, distance, required_rate):
        """选择最适合的收发器模式"""
        valid_modes = [m for m in self.params['transponder_modes']
                       if m['reach'] >= distance and m['capacity'] >= required_rate]
        if not valid_modes:
            return None

        # 选择频谱效率最高的模式(使用最少FS)
        return min(valid_modes, key=lambda x: x['fs'])

    def _check_spectrum_availability(self, u, v, required_fs):
        """检查物理路径上的频谱可用性"""
        path = nx.shortest_path(self.topology['physical'], u, v, weight='length')

        # 找到路径上所有边的连续可用频谱块
        common_available = None
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1]) if path[i] < path[i + 1] else (path[i + 1], path[i])
            used_fs = sorted(self.spectrum_allocations[edge])

            # 找到连续可用FS块
            available_blocks = []
            prev = -1
            start = 0
            for fs in used_fs:
                if fs > prev + 1:
                    available_blocks.append((start, fs - 1))
                start = fs + 1
                prev = fs
            available_blocks.append((start, self.params['total_fs'] - 1))

            # 找到足够大的连续块
            edge_available = [block for block in available_blocks
                              if block[1] - block[0] + 1 >= required_fs]

            if not edge_available:
                return False

            # 取交集
            if common_available is None:
                common_available = set()
                for block in edge_available:
                    common_available.update(range(block[0], block[0] + required_fs))
            else:
                new_available = set()
                for block in edge_available:
                    possible = range(block[0], block[0] + required_fs)
                    if any(fs in common_available for fs in possible):
                        new_available.update(fs for fs in possible if fs in common_available)
                common_available = new_available

                if not common_available:
                    return False

        return len(common_available) >= required_fs if common_available else False

    def _calculate_edge_cost(self, edge_type, u, v, demand):
        """计算边权重，严格遵循论文公式(23)"""
        # 基本参数
        c0 = 1e-5  # 基础成本
        c0_new = 1  # 新光路径基础成本
        c_new = 100  # 基于光距离的成本
        c_f = 10  # 碎片化成本
        c_prime = 0.001  # 光距离平方成本
        c_u = 1  # 数据率成本
        c_old = 1000  # 现有光路径成本

        if edge_type == 'EL':
            # 现有光路径
            lp = self.topology['lightpath'].edges[u, v]
            h = nx.shortest_path_length(self.topology['physical'], u, v, weight='length')
            return c0 + (1 - 1) * (
                        c0_new + c_new * h + c_f * 0 + c_prime * h ** 2 + c_u * 10 ** (-lp['capacity'] / 100)) + \
                (1 - 0) * (c_old * h)
        elif edge_type == 'EEL':
            # 可扩展光路径
            lp = self.topology['lightpath'].edges[u, v]
            h = nx.shortest_path_length(self.topology['physical'], u, v, weight='length')
            delta_f = self._calculate_fragmentation_change(u, v, demand['rate'])
            return c0 + 1 * (
                        c0_new + c_new * h + c_f * delta_f + c_prime * h ** 2 + c_u * 10 ** (-lp['capacity'] / 100)) + \
                (1 - 1) * (c_old * h)
        else:  # PL
            # 潜在光路径
            h = nx.shortest_path_length(self.topology['physical'], u, v, weight='length')
            mode = self._select_transponder_mode(h, demand['rate'])
            delta_f = self._calculate_fragmentation_change(u, v, demand['rate'], is_new=True)
            return c0 + 1 * (
                        c0_new + c_new * h + c_f * delta_f + c_prime * h ** 2 + c_u * 10 ** (-mode['capacity'] / 100)) + \
                (1 - 1) * (c_old * h)

    def _calculate_fragmentation_change(self, u, v, rate, is_new=False):
        """计算频谱碎片化变化(简化实现)"""
        # 实际实现需要更精确的碎片化计算
        return 0

    def _find_lightpath_path(self, cag, demand):
        """使用标签设置算法寻找光路径"""
        src, dst = demand['src'], demand['dst']
        max_hops = self.params['max_hops']
        max_otn = self.params['max_otn_switching']

        # 初始化标签
        initial_label = {
            'node': src,
            'cost': 0,
            'hops': 0,
            'path': [src],
            'used_fs': set(),
            'otn_switches': 0,
            'lightpaths': []
        }

        heap = [initial_label]
        visited = defaultdict(list)

        while heap:
            current_label = heappop(heap)

            # 检查是否到达目的地
            if current_label['node'] == dst:
                return current_label

            # 检查跳数限制
            if current_label['hops'] >= max_hops:
                continue

            # 扩展当前标签
            for neighbor in cag.neighbors(current_label['node']):
                edge_data = cag[current_label['node']][neighbor]

                # 检查容量
                if edge_data['capacity'] < demand['rate']:
                    continue

                # 创建新标签
                new_label = {
                    'node': neighbor,
                    'cost': current_label['cost'] + edge_data['cost'],
                    'hops': current_label['hops'] + 1,
                    'path': current_label['path'] + [neighbor],
                    'used_fs': current_label['used_fs'].copy(),
                    'otn_switches': current_label['otn_switches'],
                    'lightpaths': current_label['lightpaths'].copy()
                }

                # 处理不同类型的光路径
                if edge_data['type'] == 'PL':
                    # 检查频谱资源
                    fs_needed = edge_data['fs']
                    if not self._allocate_spectrum(current_label['node'], neighbor, fs_needed, new_label['used_fs']):
                        continue

                    # 记录新光路径
                    new_label['lightpaths'].append((
                        current_label['node'], neighbor,
                        edge_data['fs'], edge_data['mode']
                    ))

                elif edge_data['type'] == 'EEL':
                    # 检查是否可以扩展频谱
                    if not self._can_extend_spectrum(
                            current_label['node'], neighbor,
                            edge_data['additional'], new_label['used_fs']):
                        continue

                # 如果是中间节点，增加OTN交换计数
                if neighbor != dst and neighbor != src:
                    new_label['otn_switches'] += 1

                # 检查OTN交换容量限制
                if new_label['otn_switches'] > max_otn:
                    continue

                # 检查是否支配已有标签
                dominated = False
                for existing_label in visited[neighbor]:
                    if (existing_label['cost'] <= new_label['cost'] and
                            existing_label['hops'] <= new_label['hops'] and
                            existing_label['otn_switches'] <= new_label['otn_switches']):
                        dominated = True
                        break

                if not dominated:
                    # 移除被新标签支配的标签
                    visited[neighbor] = [label for label in visited[neighbor]
                                         if not (new_label['cost'] <= label['cost'] and
                                                 new_label['hops'] <= label['hops'] and
                                                 new_label['otn_switches'] <= label['otn_switches'])]
                    visited[neighbor].append(new_label)
                    heappush(heap, new_label)

        return None

    def _allocate_spectrum(self, u, v, fs_needed, used_fs):
        """分配频谱资源"""
        path = nx.shortest_path(self.topology['physical'], u, v, weight='length')

        # 找到所有边的公共可用连续FS块
        common_blocks = None
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1]) if path[i] < path[i + 1] else (path[i + 1], path[i])
            used = sorted(self.spectrum_allocations[edge])

            # 计算可用块
            available = []
            prev = -1
            start = 0
            for fs in used:
                if fs > prev + 1:
                    available.append((start, fs - 1))
                start = fs + 1
                prev = fs
            available.append((start, self.params['total_fs'] - 1))

            # 筛选足够大的块
            edge_blocks = [block for block in available if block[1] - block[0] + 1 >= fs_needed]

            if not edge_blocks:
                return False

            # 计算公共可用块
            if common_blocks is None:
                common_blocks = []
                for block in edge_blocks:
                    for fs in range(block[0], block[0] + fs_needed):
                        common_blocks.append(fs)
            else:
                new_common = []
                for block in edge_blocks:
                    for fs in range(block[0], block[0] + fs_needed):
                        if fs in common_blocks:
                            new_common.append(fs)
                common_blocks = new_common

                if len(common_blocks) < fs_needed:
                    return False

        # 选择第一个可用块
        if common_blocks and len(common_blocks) >= fs_needed:
            selected = common_blocks[:fs_needed]
            used_fs.update(selected)
            return True

        return False

    def _can_extend_spectrum(self, u, v, additional, used_fs):
        """检查是否可以扩展频谱"""
        # 简化实现 - 实际需要考虑现有光路径的频谱分配
        return True

    def _establish_lightpath(self, u, v, fs, mode, demand):
        """建立光路径"""
        # 分配频谱
        path = nx.shortest_path(self.topology['physical'], u, v, weight='length')
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1]) if path[i] < path[i + 1] else (path[i + 1], path[i])
            self.spectrum_allocations[edge].update(range(fs))

        # 添加/更新光路径
        if self.topology['lightpath'].has_edge(u, v):
            self.topology['lightpath'].edges[u, v]['capacity'] += mode['capacity']
            self.topology['lightpath'].edges[u, v]['fs'] = fs
            self.topology['lightpath'].edges[u, v]['spectrum'].update(range(fs))
        else:
            self.topology['lightpath'].add_edge(u, v,
                                                capacity=mode['capacity'],
                                                fs=fs,
                                                demands=[],
                                                spectrum=set(range(fs)))

        # 添加业务需求
        self.topology['lightpath'].edges[u, v]['demands'].append(demand)

        # 更新OTN交换容量
        if u != demand['src'] and u != demand['dst']:
            self.otn_switch_usage[u] += demand['rate']
        if v != demand['src'] and v != demand['dst']:
            self.otn_switch_usage[v] += demand['rate']

    def _extend_lightpath(self, u, v, additional, demand):
        """扩展光路径容量"""
        # 简化实现 - 实际需要分配额外频谱
        self.topology['lightpath'].edges[u, v]['demands'].append(demand)

        # 更新OTN交换容量
        if u != demand['src'] and u != demand['dst']:
            self.otn_switch_usage[u] += demand['rate']
        if v != demand['src'] and v != demand['dst']:
            self.otn_switch_usage[v] += demand['rate']

    def _update_metrics(self, served, blocked, total_hops, total_otn, multi_demand, utilizations):
        """更新性能指标"""
        total = served + blocked
        self.metrics['blocking_ratio'].append(blocked / total if total > 0 else 0)
        self.metrics['avg_hops'].append(total_hops / served if served > 0 else 0)
        self.metrics['otn_switching'].append(total_otn / served if served > 0 else 0)
        self.metrics['multi_demand_lightpaths'].append(multi_demand / served if served > 0 else 0)
        self.metrics['lightpath_utilization'].append(np.mean(utilizations) if utilizations else 0)

        # 计算频谱利用率
        total_used = 0
        for edge in self.topology['physical'].edges():
            total_used += len(self.spectrum_allocations[edge])
        total_available = self.params['total_fs'] * self.topology['physical'].number_of_edges()
        self.metrics['spectrum_usage'].append(total_used / total_available)

        # 计算吞吐量 - 修正后的版本
        total_rate = 0
        for u, v in self.topology['lightpath'].edges():
            total_rate += sum(d['rate'] for d in self.topology['lightpath'].edges[u, v]['demands'])
        self.metrics['throughput'].append(total_rate)

    def run_simulation(self, traffic_intensity, sim_time=1000, time_step=1):
        """运行仿真"""
        current_time = 0
        active_demands = []
        served_demands = 0
        blocked_demands = 0
        total_hops = 0
        total_otn_switches = 0
        multi_demand_count = 0
        lightpath_utilizations = []

        # 清空网络状态
        self._init_network_layers()
        self.spectrum_allocations = {edge: set() for edge in self.topology['physical'].edges()}

        progress = tqdm(total=sim_time, desc=f"ρ={traffic_intensity}erl")

        while current_time < sim_time:
            # 生成新业务
            num_new = poisson.rvs(traffic_intensity * time_step)
            for _ in range(num_new):
                demand = self._generate_demand(current_time, traffic_intensity)
                active_demands.append(demand)

            # 处理业务到达和离去
            new_blocked = 0
            new_served = 0
            new_hops = 0
            new_otn = 0
            new_multi = 0

            for demand in active_demands[:]:
                if demand['status'] == 'pending':
                    # 尝试建立连接
                    cag = self._build_cag(demand)
                    if cag is None:
                        demand['status'] = 'blocked'
                        new_blocked += 1
                        continue

                    path = self._find_lightpath_path(cag, demand)
                    if path is None:
                        demand['status'] = 'blocked'
                        new_blocked += 1
                        continue

                    # 建立连接
                    demand['status'] = 'served'
                    new_served += 1
                    new_hops += path['hops']
                    new_otn += path['otn_switches']

                    # 处理光路径
                    for i in range(len(path['path']) - 1):
                        u, v = path['path'][i], path['path'][i + 1]
                        edge_data = cag[u][v]

                        if edge_data['type'] == 'PL':
                            self._establish_lightpath(u, v, edge_data['fs'], edge_data['mode'], demand)
                        elif edge_data['type'] == 'EEL':
                            self._extend_lightpath(u, v, edge_data['additional'], demand)
                        else:  # EL
                            self.topology['lightpath'].edges[u, v]['demands'].append(demand)
                            if len(self.topology['lightpath'].edges[u, v]['demands']) > 1:
                                new_multi += 1

                            # 计算利用率
                            util = sum(d['rate'] for d in self.topology['lightpath'].edges[u, v]['demands']) / \
                                   self.topology['lightpath'].edges[u, v]['capacity']
                            lightpath_utilizations.append(util)

                # 检查业务是否完成
                if demand['end_time'] <= current_time:
                    if demand['status'] == 'served':
                        # 释放资源
                        for u, v in self.topology['lightpath'].edges():
                            if demand in self.topology['lightpath'].edges[u, v]['demands']:
                                self.topology['lightpath'].edges[u, v]['demands'].remove(demand)

                                # 更新OTN交换容量
                                if u != demand['src'] and u != demand['dst']:
                                    self.otn_switch_usage[u] -= demand['rate']
                                if v != demand['src'] and v != demand['dst']:
                                    self.otn_switch_usage[v] -= demand['rate']

                    active_demands.remove(demand)

            # 更新统计
            served_demands += new_served
            blocked_demands += new_blocked
            total_hops += new_hops
            total_otn_switches += new_otn
            multi_demand_count += new_multi

            current_time += time_step
            progress.update(time_step)

        progress.close()

        # 更新最终指标
        self._update_metrics(served_demands, blocked_demands,
                             total_hops, total_otn_switches,
                             multi_demand_count, lightpath_utilizations)

    def plot_results(self, traffic_intensities):
        """绘制结果图并保存到./results目录"""
        import os
        # 创建结果目录
        os.makedirs('./results', exist_ok=True)

        # 准备数据
        metrics = ['blocking_ratio', 'spectrum_usage', 'avg_hops',
                   'otn_switching', 'multi_demand_lightpaths', 'lightpath_utilization']
        titles = ['Blocking Ratio', 'Average Spectrum Usage', 'Average Hops per Demand',
                  'Average OTN Switching', 'Multi-Demand Lightpath Ratio', 'Lightpath Utilization']
        ylabels = ['Blocking Ratio', 'Spectrum Usage (%)', 'Average Hops',
                   'OTN Switching (Gb/s)', 'Ratio', 'Utilization']
        filenames = ['blocking_ratio.png', 'spectrum_usage.png', 'avg_hops.png',
                     'otn_switching.png', 'multi_demand.png', 'lightpath_utilization.png']

        # 绘制每个指标
        for metric, title, ylabel, filename in zip(metrics, titles, ylabels, filenames):
            plt.figure(figsize=(10, 6))
            plt.plot(traffic_intensities, self.metrics[metric][:len(traffic_intensities)],
                     marker='o', linestyle='-', linewidth=2)
            plt.xlabel('Traffic Intensity (erl)')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)

            # 保存到results目录
            save_path = os.path.join('./results', filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        # 保存原始数据为CSV文件
        self._save_metrics_to_csv(traffic_intensities)

    def _save_metrics_to_csv(self, traffic_intensities):
        """将指标数据保存为CSV文件"""
        import os
        import pandas as pd

        # 准备数据
        data = {
            'Traffic_Intensity': traffic_intensities
        }

        # 添加各个指标
        for metric in self.metrics:
            if metric != 'throughput':  # 吞吐量数据格式不同
                data[metric] = self.metrics[metric][:len(traffic_intensities)]

        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        csv_path = os.path.join('./results', 'simulation_results.csv')
        df.to_csv(csv_path, index=False)


# 主程序
if __name__ == "__main__":
    # 创建仿真器
    simulator = OpticalNetworkSimulator("topology/nsfnet.txt")

    # 运行不同业务强度的仿真
    traffic_intensities = np.arange(10, 15, 2)  # 10-20 erl, 步长2
    for intensity in traffic_intensities:
        simulator.run_simulation(intensity, sim_time=500)

    # 绘制结果
    simulator.plot_results(traffic_intensities)