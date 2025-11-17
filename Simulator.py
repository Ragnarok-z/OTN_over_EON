from Demand import Demand
import random
from Tool import *
from CAG import CAG
from tqdm import tqdm
from Defragmentation import DefragmentationEngine
import pickle


class Simulator:
    def __init__(self, network, traffic_intensity=10, num_demands=1000, random_seed=423, defrag_params={}, output_dir=None):
        self.network = network
        self.traffic_intensity = traffic_intensity
        self.num_demands = num_demands
        self.random_seed = random_seed
        self.defrag_params = defrag_params
        self.event_queue = []
        self.current_time = 0
        self.active_demands = {}
        self.blocked_demands = []
        self.completed_demands = []
        self.output_dir = output_dir

        # 用于持续收集指标的数据结构
        self.metrics_history = {
            "spectrum_usage": [],
            "avg_otn_switching": [],
            "multi_demand_lightpath_ratio": [],
            "avg_lightpath_usage": [],
            "timestamps": []
        }

        self.metrics = {
            "blocking_ratio": 0,
            "spectrum_usage": 0,
            "avg_hops": 0,
            "avg_otn_switching": 0,
            "multi_demand_lightpath_ratio": 0,
            "avg_lightpath_usage": 0
        }
        self.initialize_events()

    def initialize_events(self):
        random.seed(self.random_seed)
        # Generate arrival events
        lambda_param = self.traffic_intensity  # Arrival rate
        mu_param = 1  # Holding time rate (mean = 1/mu)

        arrival_times = []
        time = 0
        for _ in range(self.num_demands):
            inter_arrival = random.expovariate(lambda_param)
            time += inter_arrival
            arrival_times.append(time)
        # print(arrival_times[:5])

        # Create demands
        nodes = list(self.network.nodes)
        for i, arrival_time in enumerate(arrival_times):
            source, destination = random.sample(nodes, 2)
            traffic_class = random.choice(list(TrafficClass))
            demand = Demand(i, source, destination, traffic_class, arrival_time)

            # Generate holding time
            holding_time = random.expovariate(mu_param)

            demand.set_departure_time(holding_time)
            # if i<5:
            #     print(i,demand.arrival_time,demand.departure_time)

            # Add arrival and departure events
            self.event_queue.append(Event(EventType.ARRIVAL, arrival_time, demand))
            self.event_queue.append(Event(EventType.DEPARTURE, demand.departure_time, demand))

        # Sort events by time
        self.event_queue.sort()
        # print([(x.event_type,x.time,x.demand) for x in  self.event_queue[0:5]])


    def run(self, policy="MinPB", K=3, max_hops=5):
        # 记录初始状态
        self.record_metrics()

        for event in tqdm(self.event_queue):
            self.current_time = event.time
            # print(self.current_time)

            if event.event_type == EventType.ARRIVAL:
                self.process_arrival(event.demand, policy, K, max_hops)
            else:
                self.process_departure(event.demand)

            # 在每个事件处理后记录指标
            self.record_metrics()

        self.calculate_final_metrics()

    def record_metrics(self):
        """记录当前时刻的网络状态指标"""
        # 修正频谱使用率计算
        total_fs_slots = 0
        used_fs_slots = 0

        # 遍历所有物理链路（避免重复计算双向链路）
        processed_edges = set()
        for edge in self.network.fs_usage:
            # 确保每条物理链路只计算一次（忽略方向）
            physical_edge = tuple(sorted(edge))
            if physical_edge in processed_edges:
                continue
            processed_edges.add(physical_edge)

            # 每条链路有768个FS
            total_fs_slots += 768
            # 计算该链路上已使用的FS数量
            used_fs_slots += sum(1 for fs_used in self.network.fs_usage[edge] if fs_used)

        spectrum_usage = used_fs_slots / total_fs_slots if total_fs_slots > 0 else 0

        # 平均OTN交换使用率
        total_otn = sum(self.network.otn_switches[node]["used_capacity"] for node in self.network.otn_switches)
        avg_otn_per_node = total_otn / len(self.network.otn_switches) if self.network.otn_switches else 0
        avg_otn_switching = avg_otn_per_node / 24000  # As percentage of total capacity

        # 多业务光路比例
        multi_demand_lps = sum(1 for lp in self.network.lightpaths if len(lp.demands) > 1)
        multi_demand_lightpath_ratio = multi_demand_lps / len(self.network.lightpaths) if self.network.lightpaths else 0


        # total_usage = sum(lp.used_capacity / lp.capacity for lp in self.network.lightpaths)
        # avg_lightpath_usage = total_usage / len(self.network.lightpaths) if self.network.lightpaths else 0
        # 平均光路容量使用率
        total_lp_used = sum(lp.used_capacity for lp in self.network.lightpaths)
        total_lp_cap = sum(lp.capacity for lp in self.network.lightpaths)
        avg_lightpath_usage = total_lp_used / total_lp_cap if total_lp_cap else 0

        # 记录到历史数据
        self.metrics_history["spectrum_usage"].append(spectrum_usage)
        self.metrics_history["avg_otn_switching"].append(avg_otn_switching)
        self.metrics_history["multi_demand_lightpath_ratio"].append(multi_demand_lightpath_ratio)
        self.metrics_history["avg_lightpath_usage"].append(avg_lightpath_usage)
        self.metrics_history["timestamps"].append(self.current_time)

    def calculate_final_metrics(self):
        """计算最终的指标（基于整个仿真过程的平均值）"""
        # 阻塞率（保持不变）
        total_demands = len(self.completed_demands) + len(self.blocked_demands)
        self.metrics["blocking_ratio"] = len(self.blocked_demands) / total_demands if total_demands > 0 else 0

        # 平均跳数（保持不变）
        total_hops = sum(len(demand.path) - 1 for demand in self.completed_demands if demand.path)
        self.metrics["avg_hops"] = total_hops / len(self.completed_demands) if self.completed_demands else 0

        # 计算时间加权平均值
        if len(self.metrics_history["timestamps"]) > 1:
            total_time = self.metrics_history["timestamps"][-1] - self.metrics_history["timestamps"][0]

            # 频谱使用率的时间加权平均
            time_weighted_spectrum_usage = self.calculate_time_weighted_average(
                self.metrics_history["spectrum_usage"],
                self.metrics_history["timestamps"]
            )
            self.metrics["spectrum_usage"] = time_weighted_spectrum_usage

            # OTN交换使用率的时间加权平均
            time_weighted_otn_switching = self.calculate_time_weighted_average(
                self.metrics_history["avg_otn_switching"],
                self.metrics_history["timestamps"]
            )
            self.metrics["avg_otn_switching"] = time_weighted_otn_switching

            # 多业务光路比例的时间加权平均
            time_weighted_multi_demand_ratio = self.calculate_time_weighted_average(
                self.metrics_history["multi_demand_lightpath_ratio"],
                self.metrics_history["timestamps"]
            )
            self.metrics["multi_demand_lightpath_ratio"] = time_weighted_multi_demand_ratio

            # 光路使用率的时间加权平均
            time_weighted_lightpath_usage = self.calculate_time_weighted_average(
                self.metrics_history["avg_lightpath_usage"],
                self.metrics_history["timestamps"]
            )
            self.metrics["avg_lightpath_usage"] = time_weighted_lightpath_usage
        else:
            # 如果只有一个时间点，使用该点的值
            self.metrics["spectrum_usage"] = self.metrics_history["spectrum_usage"][-1] if self.metrics_history[
                "spectrum_usage"] else 0
            self.metrics["avg_otn_switching"] = self.metrics_history["avg_otn_switching"][-1] if self.metrics_history[
                "avg_otn_switching"] else 0
            self.metrics["multi_demand_lightpath_ratio"] = self.metrics_history["multi_demand_lightpath_ratio"][-1] if \
            self.metrics_history["multi_demand_lightpath_ratio"] else 0
            self.metrics["avg_lightpath_usage"] = self.metrics_history["avg_lightpath_usage"][-1] if \
            self.metrics_history["avg_lightpath_usage"] else 0

    def calculate_time_weighted_average(self, values, timestamps):
        """计算时间加权平均值"""
        if len(values) < 2:
            return values[0] if values else 0

        total_weighted_sum = 0
        total_time = 0

        for i in range(1, len(values)):
            time_interval = timestamps[i] - timestamps[i - 1]
            avg_value = (values[i - 1] + values[i]) / 2  # 使用梯形法则
            total_weighted_sum += avg_value * time_interval
            total_time += time_interval

        return total_weighted_sum / total_time if total_time > 0 else 0

    # 原有的 process_arrival, process_departure 等方法保持不变
    def process_arrival(self, demand, policy, K, max_hops):
        # Build CAG for this demand
        cag = CAG(self.network, demand, K)

        # with open(f'{self.output_dir}/damand{demand.id}_CAG.pkl', 'wb') as f:
        #     pickle.dump(cag, f)

        # Find shortest path using label-setting algorithm

        path = cag.find_shortest_path(policy, max_hops)

        # for i in range(10):
        #     path_ = cag.find_shortest_path(policy, max_hops)
        #     assert path == path_, f'shortest path failed: {i,demand.id,path,path_}'

        if not path:
            self.blocked_demands.append(demand)
            if self.defrag_params["en"]:
                defrag_engine = DefragmentationEngine(self.network)
                # 执行碎片整理
                defrag_engine.trigger_defragmentation(demand)
            # print(demand.id, "is blocked")
            return

        # Allocate resources along the path
        lightpaths_used = []
        otn_switching_nodes = set()

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge_info = cag.edges[u][v]

            if edge_info["type"] == "EL":
                # Use existing lightpath
                lightpath = edge_info["lightpath"]
                lightpath.add_demand(demand)
                lightpaths_used.append(lightpath)

                # # Update OTN switching if this is an intermediate node
                # if i > 0:
                #     otn_switching_nodes.add(u)
                # 将所有节点都加入OTN switching节点使用集合
                if i==0:
                    otn_switching_nodes.add(u)
                otn_switching_nodes.add(v)
            elif edge_info["type"] == "EEL":
                # Extend existing lightpath (simplified - just use it)
                # completed
                # print("12312132")
                eel = edge_info["lightpath"]
                o_lp = eel["original_lightpath"]
                e_lp = eel["extended_lightpath"]
                e_lightpath = edge_info["lightpath"]["extended_lightpath"]
                self.network.remove_lightpath(o_lp)
                new_lp = self.network.create_lightpath(e_lp.source, e_lp.destination, e_lp.transponder_mode, e_lp.fs_allocated, e_lp.path_in_G0)
                for d in o_lp.demands:
                    new_lp.add_demand(d)
                    index = d.lightpaths_used.index(o_lp)
                    d.lightpaths_used[index] = new_lp

                new_lp.add_demand(demand)
                lightpaths_used.append(new_lp)

                # # Update OTN switching if this is an intermediate node
                # if i > 0:
                #     otn_switching_nodes.add(u)
                # 将所有节点都加入OTN switching节点使用集合
                if i==0:
                    otn_switching_nodes.add(u)
                otn_switching_nodes.add(v)
            elif edge_info["type"] == "PL":
                # Create new lightpath
                transponder_mode = edge_info["transponder_mode"]
                path_G0 = edge_info["path_G0"]
                fs_block = edge_info["fs_block"]

                lightpath = self.network.create_lightpath(u, v, transponder_mode, fs_block, path_G0)
                lightpath.add_demand(demand)
                lightpaths_used.append(lightpath)

                # # Update OTN switching if this is an intermediate node
                # if i > 0:
                #     otn_switching_nodes.add(u)
                # 将所有节点都加入OTN switching节点使用集合
                if i==0:
                    otn_switching_nodes.add(u)
                otn_switching_nodes.add(v)

        # Update OTN switching capacity
        for node in otn_switching_nodes:
            self.network.update_otn_switching(node, demand.traffic_class.value)

        # Record the path and lightpaths used
        demand.path = path
        demand.lightpaths_used = lightpaths_used
        self.active_demands[demand.id] = demand
        # with open(f"{self.output_dir}/request_results.txt", 'a', encoding='utf-8') as f:
        #     f.write(f"Demand {demand.id} is served, traffic_class: {demand.traffic_class},G1_path: {demand.path}\n")
        #     for lp in lightpaths_used:
        #         f.write(f"  lp_sd: {(lp.source, lp.destination)} \n    path_in_G0: {lp.path_in_G0} \n    fs_allocated: {lp.fs_allocated}\n")

    def process_departure(self, demand):
        if demand.id not in self.active_demands:
            return  # Demand was blocked

        # Release resources
        for lightpath in demand.lightpaths_used:
            lightpath.remove_demand(demand)

            # Remove lightpath if it's no longer carrying any demands
            if len(lightpath.demands) == 0 and lightpath in self.network.lightpaths:
                self.network.remove_lightpath(lightpath)

        # Update OTN switching capacity for intermediate nodes
        # 将起终点也加入释放的列表中
        if demand.path:
            otn_switching_nodes = set()
            for i in range(0, len(demand.path)):
                otn_switching_nodes.add(demand.path[i])

            for node in otn_switching_nodes:
                self.network.update_otn_switching(node, -demand.traffic_class.value)

        # Record completed demand
        self.completed_demands.append(demand)
        del self.active_demands[demand.id]

    def get_metrics(self):
        return self.metrics

# from Demand import Demand
# import random
# from Tool import *
# from CAG import CAG
# from tqdm import tqdm
# from Defragmentation import DefragmentationEngine
#
#
# class Simulator:
#     def __init__(self, network, traffic_intensity=10, num_demands=1000, random_seed=423, defrag_params={}):
#         self.network = network
#         self.traffic_intensity = traffic_intensity
#         self.num_demands = num_demands
#         self.random_seed = random_seed
#         self.defrag_params = defrag_params
#         self.event_queue = []
#         self.current_time = 0
#         self.active_demands = {}
#         self.blocked_demands = []
#         self.completed_demands = []
#         self.metrics = {
#             "blocking_ratio": 0,
#             "spectrum_usage": 0,
#             "avg_hops": 0,
#             "avg_otn_switching": 0,
#             "multi_demand_lightpath_ratio": 0,
#             "avg_lightpath_usage": 0
#         }
#         self.initialize_events()
#
#     def initialize_events(self):
#         random.seed(self.random_seed)
#         # Generate arrival events
#         lambda_param = self.traffic_intensity  # Arrival rate
#         mu_param = 1  # Holding time rate (mean = 1/mu)
#
#         arrival_times = []
#         time = 0
#         for _ in range(self.num_demands):
#             inter_arrival = random.expovariate(lambda_param)
#             time += inter_arrival
#             arrival_times.append(time)
#
#         # Create demands
#         nodes = list(self.network.nodes)
#         for i, arrival_time in enumerate(arrival_times):
#             source, destination = random.sample(nodes, 2)
#             traffic_class = random.choice(list(TrafficClass))
#             demand = Demand(i, source, destination, traffic_class, arrival_time)
#
#             # Generate holding time
#             holding_time = random.expovariate(mu_param)
#             demand.set_departure_time(holding_time)
#
#             # Add arrival and departure events
#             self.event_queue.append(Event(EventType.ARRIVAL, arrival_time, demand))
#             self.event_queue.append(Event(EventType.DEPARTURE, demand.departure_time, demand))
#
#         # Sort events by time
#         self.event_queue.sort()
#
#     def run(self, policy="MinPB", K=3, max_hops=5):
#         for event in tqdm(self.event_queue):
#             self.current_time = event.time
#
#             if event.event_type == EventType.ARRIVAL:
#                 self.process_arrival(event.demand, policy, K, max_hops)
#             else:
#                 self.process_departure(event.demand)
#
#         self.calculate_metrics()
#
#     def process_arrival(self, demand, policy, K, max_hops):
#         # Build CAG for this demand
#         cag = CAG(self.network, demand, K)
#
#         # Find shortest path using label-setting algorithm
#         path = cag.find_shortest_path(policy, max_hops)
#
#         if not path:
#             self.blocked_demands.append(demand)
#             if self.defrag_params["en"]:
#                 defrag_engine = DefragmentationEngine(self.network)
#                 # 执行碎片整理
#                 defrag_engine.trigger_defragmentation(demand)
#             return
#
#         # Allocate resources along the path
#         lightpaths_used = []
#         otn_switching_nodes = set()
#
#         for i in range(len(path) - 1):
#             u = path[i]
#             v = path[i + 1]
#             edge_info = cag.edges[u][v]
#
#             if edge_info["type"] == "EL":
#                 # Use existing lightpath
#                 lightpath = edge_info["lightpath"]
#                 lightpath.add_demand(demand)
#                 lightpaths_used.append(lightpath)
#
#                 # Update OTN switching if this is an intermediate node
#                 if i > 0:
#                     otn_switching_nodes.add(u)
#             elif edge_info["type"] == "EEL":
#                 # Extend existing lightpath (simplified - just use it)
#                 # completed
#                 # print("12312132")
#                 eel = edge_info["lightpath"]
#                 o_lp = eel["original_lightpath"]
#                 e_lp = eel["extended_lightpath"]
#                 e_lightpath = edge_info["lightpath"]["extended_lightpath"]
#                 self.network.remove_lightpath(o_lp)
#                 new_lp = self.network.create_lightpath(e_lp.source, e_lp.destination, e_lp.transponder_mode, e_lp.fs_allocated, e_lp.path_in_G0)
#                 for d in o_lp.demands:
#                     new_lp.add_demand(d)
#                     index = d.lightpaths_used.index(o_lp)
#                     d.lightpaths_used[index] = new_lp
#
#                 new_lp.add_demand(demand)
#                 lightpaths_used.append(new_lp)
#
#                 if i > 0:
#                     otn_switching_nodes.add(u)
#             elif edge_info["type"] == "PL":
#                 # Create new lightpath
#                 transponder_mode = edge_info["transponder_mode"]
#                 path_G0 = edge_info["path_G0"]
#                 fs_block = edge_info["fs_block"]
#
#                 lightpath = self.network.create_lightpath(u, v, transponder_mode, fs_block, path_G0)
#                 lightpath.add_demand(demand)
#                 lightpaths_used.append(lightpath)
#
#                 if i > 0:
#                     otn_switching_nodes.add(u)
#
#         # Update OTN switching capacity
#         for node in otn_switching_nodes:
#             self.network.update_otn_switching(node, demand.traffic_class.value)
#
#         # Record the path and lightpaths used
#         demand.path = path
#         demand.lightpaths_used = lightpaths_used
#         self.active_demands[demand.id] = demand
#
#     def process_departure(self, demand):
#         if demand.id not in self.active_demands:
#             return  # Demand was blocked
#
#         # Release resources
#         for lightpath in demand.lightpaths_used:
#             lightpath.remove_demand(demand)
#
#             # Remove lightpath if it's no longer carrying any demands
#             if len(lightpath.demands) == 0 and lightpath in self.network.lightpaths:
#                 self.network.remove_lightpath(lightpath)
#
#         # Update OTN switching capacity for intermediate nodes
#         if demand.path:
#             otn_switching_nodes = set()
#             for i in range(1, len(demand.path) - 1):
#                 otn_switching_nodes.add(demand.path[i])
#
#             for node in otn_switching_nodes:
#                 self.network.update_otn_switching(node, -demand.traffic_class.value)
#
#         # Record completed demand
#         self.completed_demands.append(demand)
#         del self.active_demands[demand.id]
#
#     def calculate_metrics(self):
#         # Blocking ratio
#         total_demands = len(self.completed_demands) + len(self.blocked_demands)
#         self.metrics["blocking_ratio"] = len(self.blocked_demands) / total_demands if total_demands > 0 else 0
#
#         # Spectrum usage
#         total_fs = 768 * len(self.network.edges)  # Total FS in network
#         used_fs = sum(sum(self.network.fs_usage[edge]) for edge in self.network.fs_usage)
#         self.metrics["spectrum_usage"] = used_fs / total_fs if total_fs > 0 else 0
#
#         # Average hops per demand
#         total_hops = sum(len(demand.path) - 1 for demand in self.completed_demands if demand.path)
#         self.metrics["avg_hops"] = total_hops / len(self.completed_demands) if self.completed_demands else 0
#
#         # Average OTN switching
#         total_otn = sum(self.network.otn_switches[node]["used_capacity"] for node in self.network.otn_switches)
#         avg_otn_per_node = total_otn / len(self.network.otn_switches) if self.network.otn_switches else 0
#         self.metrics["avg_otn_switching"] = avg_otn_per_node / 24000  # As percentage of total capacity
#
#         # Multi-demand lightpath ratio
#         multi_demand_lps = sum(1 for lp in self.network.lightpaths if len(lp.demands) > 1)
#         self.metrics["multi_demand_lightpath_ratio"] = multi_demand_lps / len(
#             self.network.lightpaths) if self.network.lightpaths else 0
#
#         # Average lightpath capacity usage
#         total_usage = sum(lp.used_capacity / lp.capacity for lp in self.network.lightpaths)
#         self.metrics["avg_lightpath_usage"] = total_usage / len(
#             self.network.lightpaths) if self.network.lightpaths else 0
#
#     def get_metrics(self):
#         return self.metrics