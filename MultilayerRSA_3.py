import heapq
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from enum import Enum
import json
import os


# Constants and parameters
class TrafficClass(Enum):
    GE10 = 10
    GE100 = 100
    GE400 = 400


# Transponder operational modes (from Table V in the paper)
TRANSPONDER_MODES_original = [
    {"capacity": 200, "bitrate": 200, "baudrate": 95, "fs_required": 19, "max_spans": 125},
    {"capacity": 300, "bitrate": 300, "baudrate": 95, "fs_required": 19, "max_spans": 88},
    {"capacity": 400, "bitrate": 400, "baudrate": 95, "fs_required": 19, "max_spans": 54},
    {"capacity": 500, "bitrate": 500, "baudrate": 95, "fs_required": 19, "max_spans": 35},
    {"capacity": 600, "bitrate": 600, "baudrate": 95, "fs_required": 19, "max_spans": 18},
    {"capacity": 700, "bitrate": 700, "baudrate": 95, "fs_required": 19, "max_spans": 9},
    {"capacity": 800, "bitrate": 800, "baudrate": 95, "fs_required": 19, "max_spans": 4},
    {"capacity": 100, "bitrate": 100, "baudrate": 56, "fs_required": 12, "max_spans": 130},
    {"capacity": 200, "bitrate": 200, "baudrate": 56, "fs_required": 12, "max_spans": 61},
    {"capacity": 300, "bitrate": 300, "baudrate": 56, "fs_required": 12, "max_spans": 34},
    {"capacity": 400, "bitrate": 400, "baudrate": 56, "fs_required": 12, "max_spans": 10},
    {"capacity": 100, "bitrate": 100, "baudrate": 35, "fs_required": 8, "max_spans": 75},
    {"capacity": 200, "bitrate": 200, "baudrate": 35, "fs_required": 8, "max_spans": 16}
]

# selects the transponder operational mode that minimizes spectral usage while maximizes data rate
TRANSPONDER_MODES = [
    {"capacity": 200, "bitrate": 200, "baudrate": 35, "fs_required": 8, "max_spans": 16},
    {"capacity": 100, "bitrate": 100, "baudrate": 35, "fs_required": 8, "max_spans": 75},
    {"capacity": 400, "bitrate": 400, "baudrate": 56, "fs_required": 12, "max_spans": 10},
    {"capacity": 300, "bitrate": 300, "baudrate": 56, "fs_required": 12, "max_spans": 34},
    {"capacity": 200, "bitrate": 200, "baudrate": 56, "fs_required": 12, "max_spans": 61},
    {"capacity": 100, "bitrate": 100, "baudrate": 56, "fs_required": 12, "max_spans": 130},
    {"capacity": 800, "bitrate": 800, "baudrate": 95, "fs_required": 19, "max_spans": 4},
    {"capacity": 700, "bitrate": 700, "baudrate": 95, "fs_required": 19, "max_spans": 9},
    {"capacity": 600, "bitrate": 600, "baudrate": 95, "fs_required": 19, "max_spans": 18},
    {"capacity": 500, "bitrate": 500, "baudrate": 95, "fs_required": 19, "max_spans": 35},
    {"capacity": 400, "bitrate": 400, "baudrate": 95, "fs_required": 19, "max_spans": 54},
    {"capacity": 300, "bitrate": 300, "baudrate": 95, "fs_required": 19, "max_spans": 88},
    {"capacity": 200, "bitrate": 200, "baudrate": 95, "fs_required": 19, "max_spans": 125},
]

# Traffic engineering policy coefficients (from Table IV in the paper)
POLICY_COEFFICIENTS = {
    "MinEn": {"c0": 100000, "c0_new": 10000, "c_new": 1000, "cf": 10, "c_prime": 0, "cu": 0, "c_old": 1000},
    "MaxMux": {"c0": 0, "c0_new": 1, "c_new": 100, "cf": 0, "c_prime": 0, "cu": 0, "c_old": 0},
    "MaxSE": {"c0": 0, "c0_new": 0, "c_new": 1, "cf": 0, "c_prime": 0, "cu": 1, "c_old": 1},
    "MinPB": {"c0": 0.000001, "c0_new": 0, "c_new": 1000, "cf": 0, "c_prime": 0.001, "cu": 0, "c_old": 1000}
}


class EventType(Enum):
    ARRIVAL = 1
    DEPARTURE = 2


class Event:
    def __init__(self, event_type, time, demand=None):
        self.event_type = event_type
        self.time = time
        self.demand = demand

    def __lt__(self, other):
        return self.time < other.time


class Demand:
    def __init__(self, id, source, destination, traffic_class, arrival_time):
        self.id = id
        self.source = source
        self.destination = destination
        self.traffic_class = traffic_class
        self.arrival_time = arrival_time
        self.departure_time = None
        self.path = None
        self.lightpaths_used = []
        self.required_slots = self.calculate_required_slots()

    def calculate_required_slots(self):
        # Calculate required OTN timeslots based on traffic class
        if self.traffic_class == TrafficClass.GE10:
            return 1  # Simplified - actual calculation would be based on OTN timeslot size
        elif self.traffic_class == TrafficClass.GE100:
            return 10
        elif self.traffic_class == TrafficClass.GE400:
            return 40
        return 1

    def set_departure_time(self, holding_time):
        self.departure_time = self.arrival_time + holding_time


class Lightpath:
    def __init__(self, source, destination, transponder_mode, fs_allocated, path_in_G0):
        self.source = source
        self.destination = destination
        self.transponder_mode = transponder_mode
        self.fs_allocated = fs_allocated  # List of frequency slots
        self.path_in_G0 = path_in_G0  # Physical path in G0 layer
        self.capacity = transponder_mode["capacity"]
        self.used_capacity = 0
        self.demands = []  # List of demands using this lightpath

    def can_accommodate(self, demand):
        return (self.used_capacity + demand.traffic_class.value) <= self.capacity

    def remaining_capacity(self):
        return self.capacity - self.used_capacity

    def add_demand(self, demand):
        if self.can_accommodate(demand):
            self.demands.append(demand)
            self.used_capacity += demand.traffic_class.value
            return True
        return False

    def remove_demand(self, demand):
        if demand in self.demands:
            self.demands.remove(demand)
            self.used_capacity -= demand.traffic_class.value
            return True
        return False


class Network:
    def __init__(self, topology_file):
        self.nodes = set()
        self.edges = defaultdict(dict)  # G0 layer edges
        self.lightpaths = []  # G1 layer lightpaths
        self.otn_switches = {}  # OTN switching capacity at each node
        self.fs_usage = {}  # Frequency slot usage on each G0 edge
        self.load_topology(topology_file)
        self.initialize_network()

    def load_topology(self, topology_file):
        with open(topology_file, 'r') as f:
            n, m = map(int, f.readline().split())
            for _ in range(m):
                s, t, d = map(int, f.readline().split())
                self.nodes.add(s)
                self.nodes.add(t)
                self.edges[s][t] = d
                self.edges[t][s] = d  # Assuming undirected edges
                # Initialize FS usage for each edge
                self.fs_usage[(s, t)] = [False] * 768  # 768 FSs as per the paper
                self.fs_usage[(t, s)] = [False] * 768

    def initialize_network(self):
        # Initialize OTN switching capacity (24,000 Gb/s as per the paper)
        for node in self.nodes:
            self.otn_switches[node] = {"total_capacity": 24000, "used_capacity": 0}

    def find_k_shortest_paths(self, source, destination, k):
        # Implement Yen's algorithm to find k shortest paths
        paths = []
        heap = []

        # First find the shortest path
        shortest_path = self.dijkstra(source, destination)
        if not shortest_path:
            return paths
        paths.append(shortest_path)
        heapq.heappush(heap, (self.path_length(shortest_path), 0, shortest_path))

        for i in range(1, k):
            if not heap:
                break

            _, _, prev_path = heapq.heappop(heap)

            for j in range(len(prev_path) - 1):
                spur_node = prev_path[j]
                root_path = prev_path[:j + 1]

                # Remove edges that are part of the root path
                removed_edges = []
                for path in paths:
                    if len(path) > j and root_path == path[:j + 1]:
                        u = path[j]
                        v = path[j + 1] if j + 1 < len(path) else None
                        if v:
                            if v in self.edges[u]:
                                removed_edges.append((u, v, self.edges[u][v]))
                                del self.edges[u][v]
                            if u in self.edges[v]:
                                removed_edges.append((v, u, self.edges[v][u]))
                                del self.edges[v][u]

                # Find spur path from spur node to destination
                spur_path = self.dijkstra(spur_node, destination)

                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    if total_path not in paths:
                        heapq.heappush(heap, (self.path_length(total_path), i, total_path))

                # Restore removed edges
                for u, v, d in removed_edges:
                    self.edges[u][v] = d
                    self.edges[v][u] = d

        # Collect the k shortest paths
        result = []
        while heap and len(result) < k:
            _, _, path = heapq.heappop(heap)
            if path not in result:
                result.append(path)

        return result

    def dijkstra(self, source, destination):
        if source == destination:
            return [source]

        distances = {node: float('inf') for node in self.nodes}
        previous = {node: None for node in self.nodes}
        distances[source] = 0
        heap = [(0, source)]

        while heap:
            current_dist, current_node = heapq.heappop(heap)

            if current_node == destination:
                break

            if current_dist > distances[current_node]:
                continue

            for neighbor, distance in self.edges[current_node].items():
                new_dist = current_dist + distance
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))

        if previous[destination] is None:
            return None

        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous[current]
        return path[::-1]

    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.edges[path[i]][path[i + 1]]
        return length

    def find_available_fs_block(self, path, required_fs):
        # Find contiguous FS block along the entire path
        # Returns (start_fs, end_fs) or None if not available

        # Get all edges in the path
        edges_in_path = []
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edges_in_path.append((u, v))

        # Check each possible FS block
        for start_fs in range(768 - required_fs + 1):
            end_fs = start_fs + required_fs - 1
            available = True

            for edge in edges_in_path:
                for fs in range(start_fs, end_fs + 1):
                    if self.fs_usage[edge][fs]:
                        available = False
                        break
                if not available:
                    break

            if available:
                return (start_fs, end_fs)

        return None

    def allocate_fs_block(self, path, fs_block):
        start_fs, end_fs = fs_block
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for fs in range(start_fs, end_fs + 1):
                self.fs_usage[(u, v)][fs] = True
                self.fs_usage[(v, u)][fs] = True

    def release_fs_block(self, path, fs_block):
        start_fs, end_fs = fs_block
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for fs in range(start_fs, end_fs + 1):
                self.fs_usage[(u, v)][fs] = False
                self.fs_usage[(v, u)][fs] = False

    def find_existing_lightpaths(self, source, destination):
        # Find existing lightpaths between source and destination with available capacity
        return [lp for lp in self.lightpaths
                if lp.source == source and lp.destination == destination
                and lp.used_capacity < lp.capacity]

    def find_extendable_lightpaths(self, source, destination, demand):
        # Find lightpaths that can be extended to accommodate the demand
        # This is a simplified version - actual implementation would need to check FS availability
        return []

    def can_create_lightpath(self, source, destination, demand):
        # Check if we can create a new lightpath between source and destination
        # Find the shortest path in G0
        path_G0 = self.dijkstra(source, destination)
        if not path_G0:
            return False

        # Find appropriate transponder mode
        required_capacity = demand.traffic_class.value
        path_length = self.path_length(path_G0)

        # selects the transponder operational mode that minimizes spectral usage while maximizes data rate
        for mode in TRANSPONDER_MODES:
            if mode["capacity"] >= required_capacity and path_length <= mode["max_spans"]:
                # Check FS availability
                fs_block = self.find_available_fs_block(path_G0, mode["fs_required"])
                if fs_block:
                    return True, mode, path_G0, fs_block

        return False, None, None, None

    def create_lightpath(self, source, destination, transponder_mode, fs_block, path_G0):
        # Create a new lightpath
        lightpath = Lightpath(source, destination, transponder_mode, fs_block, path_G0)
        self.lightpaths.append(lightpath)
        self.allocate_fs_block(path_G0, fs_block)
        return lightpath

    def remove_lightpath(self, lightpath):
        if lightpath in self.lightpaths:
            self.lightpaths.remove(lightpath)
            self.release_fs_block(lightpath.path_in_G0, lightpath.fs_allocated)
            return True
        return False

    def update_otn_switching(self, node, capacity_change):
        # Update OTN switching capacity usage at a node
        if node in self.otn_switches:
            self.otn_switches[node]["used_capacity"] += capacity_change
            if self.otn_switches[node]["used_capacity"] < 0:
                self.otn_switches[node]["used_capacity"] = 0

    def get_otn_switching_usage(self, node):
        if node in self.otn_switches:
            return self.otn_switches[node]["used_capacity"]
        return 0


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
        self.nodes.add(self.demand.source)
        self.nodes.add(self.demand.destination)

        for path in k_shortest_paths:
            for node in path:
                if node != self.demand.source and node != self.demand.destination:
                    if self.network.otn_switches[node]["used_capacity"] < self.network.otn_switches[node]["total_capacity"]:
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

    def calculate_edge_weight(self, u, v, policy):
        edge_info = self.edges[u][v]
        coeffs = POLICY_COEFFICIENTS[policy]

        if edge_info["type"] == "EL" or edge_info["type"] == "EEL":
            lightpath = edge_info["lightpath"]
            h = len(lightpath.path_in_G0) - 1  # Number of hops
            return coeffs["c0"] + (1 - 1) * (coeffs["c_old"] * h)
        else:  # PL
            mode = edge_info["transponder_mode"]
            path_G0 = edge_info["path_G0"]
            h = len(path_G0) - 1
            delta_f = 0  # Simplified - actual would calculate fragmentation change
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
        heapq.heappush(priority_queue, (initial_label["cost"], id(initial_label), initial_label))

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
                    heapq.heappush(priority_queue, (new_label["cost"], id(new_label), new_label))

        return None  # No path found


class Simulator:
    def __init__(self, network, traffic_intensity=10, num_demands=1000):
        self.network = network
        self.traffic_intensity = traffic_intensity
        self.num_demands = num_demands
        self.event_queue = []
        self.current_time = 0
        self.active_demands = {}
        self.blocked_demands = []
        self.completed_demands = []
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
        # Generate arrival events
        lambda_param = self.traffic_intensity  # Arrival rate
        mu_param = 1  # Holding time rate (mean = 1/mu)

        arrival_times = []
        time = 0
        for _ in range(self.num_demands):
            inter_arrival = random.expovariate(lambda_param)
            time += inter_arrival
            arrival_times.append(time)

        # Create demands
        nodes = list(self.network.nodes)
        for i, arrival_time in enumerate(arrival_times):
            source, destination = random.sample(nodes, 2)
            traffic_class = random.choice(list(TrafficClass))
            demand = Demand(i, source, destination, traffic_class, arrival_time)

            # Generate holding time
            holding_time = random.expovariate(mu_param)
            demand.set_departure_time(holding_time)

            # Add arrival and departure events
            self.event_queue.append(Event(EventType.ARRIVAL, arrival_time, demand))
            self.event_queue.append(Event(EventType.DEPARTURE, demand.departure_time, demand))

        # Sort events by time
        self.event_queue.sort()

    def run(self, policy="MinPB", K=3, max_hops=5):
        for event in self.event_queue:
            self.current_time = event.time

            if event.event_type == EventType.ARRIVAL:
                self.process_arrival(event.demand, policy, K, max_hops)
            else:
                self.process_departure(event.demand)

        self.calculate_metrics()

    def process_arrival(self, demand, policy, K, max_hops):
        # Build CAG for this demand
        cag = CAG(self.network, demand, K)

        # Find shortest path using label-setting algorithm
        path = cag.find_shortest_path(policy, max_hops)

        if not path:
            self.blocked_demands.append(demand)
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

                # Update OTN switching if this is an intermediate node
                if i > 0:
                    otn_switching_nodes.add(u)
            elif edge_info["type"] == "EEL":
                # Extend existing lightpath (simplified - just use it)
                lightpath = edge_info["lightpath"]
                lightpath.add_demand(demand)
                lightpaths_used.append(lightpath)

                if i > 0:
                    otn_switching_nodes.add(u)
            elif edge_info["type"] == "PL":
                # Create new lightpath
                transponder_mode = edge_info["transponder_mode"]
                path_G0 = edge_info["path_G0"]
                fs_block = edge_info["fs_block"]

                lightpath = self.network.create_lightpath(u, v, transponder_mode, fs_block, path_G0)
                lightpath.add_demand(demand)
                lightpaths_used.append(lightpath)

                if i > 0:
                    otn_switching_nodes.add(u)

        # Update OTN switching capacity
        for node in otn_switching_nodes:
            self.network.update_otn_switching(node, demand.traffic_class.value)

        # Record the path and lightpaths used
        demand.path = path
        demand.lightpaths_used = lightpaths_used
        self.active_demands[demand.id] = demand

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
        if demand.path:
            otn_switching_nodes = set()
            for i in range(1, len(demand.path) - 1):
                otn_switching_nodes.add(demand.path[i])

            for node in otn_switching_nodes:
                self.network.update_otn_switching(node, -demand.traffic_class.value)

        # Record completed demand
        self.completed_demands.append(demand)
        del self.active_demands[demand.id]

    def calculate_metrics(self):
        # Blocking ratio
        total_demands = len(self.completed_demands) + len(self.blocked_demands)
        self.metrics["blocking_ratio"] = len(self.blocked_demands) / total_demands if total_demands > 0 else 0

        # Spectrum usage
        total_fs = 768 * len(self.network.edges)  # Total FS in network
        used_fs = sum(sum(self.network.fs_usage[edge]) for edge in self.network.fs_usage)
        self.metrics["spectrum_usage"] = used_fs / total_fs if total_fs > 0 else 0

        # Average hops per demand
        total_hops = sum(len(demand.path) - 1 for demand in self.completed_demands if demand.path)
        self.metrics["avg_hops"] = total_hops / len(self.completed_demands) if self.completed_demands else 0

        # Average OTN switching
        total_otn = sum(self.network.otn_switches[node]["used_capacity"] for node in self.network.otn_switches)
        avg_otn_per_node = total_otn / len(self.network.otn_switches) if self.network.otn_switches else 0
        self.metrics["avg_otn_switching"] = avg_otn_per_node / 24000  # As percentage of total capacity

        # Multi-demand lightpath ratio
        multi_demand_lps = sum(1 for lp in self.network.lightpaths if len(lp.demands) > 1)
        self.metrics["multi_demand_lightpath_ratio"] = multi_demand_lps / len(
            self.network.lightpaths) if self.network.lightpaths else 0

        # Average lightpath capacity usage
        total_usage = sum(lp.used_capacity / lp.capacity for lp in self.network.lightpaths)
        self.metrics["avg_lightpath_usage"] = total_usage / len(
            self.network.lightpaths) if self.network.lightpaths else 0

    def get_metrics(self):
        return self.metrics


def run_experiments(topology_file, output_dir="results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load network topology
    network = Network(topology_file)

    # Define traffic intensities (Erlang)
    traffic_intensities = range(10, 21, 2)

    # Define number of demand
    num_demands = 1000

    # Define policies to test
    policies = ["MinEn", "MaxMux", "MaxSE", "MinPB"]

    # Initialize results storage
    results = {policy: {
        "blocking_ratio": [],
        "spectrum_usage": [],
        "avg_hops": [],
        "avg_otn_switching": [],
        "multi_demand_lightpath_ratio": [],
        "avg_lightpath_usage": []
    } for policy in policies}

    # Run simulations for each traffic intensity and policy
    for intensity in traffic_intensities:
        print(f"Running simulations for traffic intensity: {intensity} Erlang")

        for policy in policies:
            print(f"  Testing policy: {policy}")

            # Create a new simulator for each run to ensure clean state
            simulator = Simulator(Network(topology_file), intensity, num_demands)
            simulator.run(policy)

            # Store results
            metrics = simulator.get_metrics()
            for metric in metrics:
                results[policy][metric].append(metrics[metric])

    # Save results to JSON files
    for policy in policies:
        with open(f"{output_dir}/{policy}_results.json", 'w') as f:
            json.dump(results[policy], f, indent=2)

    # Generate plots similar to those in the paper

    # Figure 4: Blocking ratio comparison
    plt.figure(figsize=(10, 6))
    for policy in policies:
        plt.plot(traffic_intensities, results[policy]["blocking_ratio"],
                 label=policy, marker='o')
    plt.xlabel("Traffic Intensity (Erlang)")
    plt.ylabel("Blocking Ratio")
    plt.title("Comparison of Blocking Ratio Across Policies")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/blocking_ratio_comparison.png")
    plt.close()

    # Figure 5: Spectrum usage
    plt.figure(figsize=(10, 6))
    for policy in policies:
        plt.plot(traffic_intensities, results[policy]["spectrum_usage"],
                 label=policy, marker='o')
    plt.xlabel("Traffic Intensity (Erlang)")
    plt.ylabel("Average Spectrum Usage (%)")
    plt.title("Spectrum Usage by Different Policies")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/spectrum_usage_comparison.png")
    plt.close()

    # Figure 6: Average hops per demand
    plt.figure(figsize=(10, 6))
    for policy in policies:
        plt.plot(traffic_intensities, results[policy]["avg_hops"],
                 label=policy, marker='o')
    plt.xlabel("Traffic Intensity (Erlang)")
    plt.ylabel("Average Hops per Demand")
    plt.title("Average Hops per Demand Across Policies")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/avg_hops_comparison.png")
    plt.close()

    # Figure 7: OTN switching usage
    plt.figure(figsize=(10, 6))
    for policy in policies:
        plt.plot(traffic_intensities, results[policy]["avg_otn_switching"],
                 label=policy, marker='o')
    plt.xlabel("Traffic Intensity (Erlang)")
    plt.ylabel("Average OTN Switching Usage (%)")
    plt.title("Impact of Policies on OTN Switching Usage")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/otn_switching_comparison.png")
    plt.close()

    # Figure 8: Multi-demand lightpath ratio
    plt.figure(figsize=(10, 6))
    for policy in policies:
        plt.plot(traffic_intensities, results[policy]["multi_demand_lightpath_ratio"],
                 label=policy, marker='o')
    plt.xlabel("Traffic Intensity (Erlang)")
    plt.ylabel("Ratio of Multi-Demand Lightpaths")
    plt.title("Multi-Demand Lightpath Ratio Across Policies")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/multi_demand_lightpath_ratio.png")
    plt.close()

    # Figure 9: Lightpath capacity utilization
    plt.figure(figsize=(10, 6))
    for policy in policies:
        plt.plot(traffic_intensities, results[policy]["avg_lightpath_usage"],
                 label=policy, marker='o')
    plt.xlabel("Traffic Intensity (Erlang)")
    plt.ylabel("Average Lightpath Capacity Utilization")
    plt.title("Lightpath Capacity Utilization Across Policies")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/lightpath_utilization_comparison.png")
    plt.close()

    print(f"All results and plots saved to {output_dir} directory")


# Example usage
if __name__ == "__main__":
    # In practice, you would have a real topology file
    topology_file = 'topology/nsfnet.txt'
    run_experiments(topology_file=topology_file, output_dir='results')