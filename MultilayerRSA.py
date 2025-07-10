import heapq
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import numpy as np

class NetworkNode:
    def __init__(self, node_id: int, otn_switching_capacity: float):
        self.node_id = node_id
        self.otn_switching_capacity = otn_switching_capacity  # in Gb/s
        self.used_otn_switching = 0.0  # currently used switching capacity


class FiberLink:
    def __init__(self, source: int, dest: int, length: float, total_fs: int = 768):
        self.source = source
        self.dest = dest
        self.length = length  # in km
        self.total_fs = total_fs
        self.used_fs = set()  # set of used frequency slots
        self.available_fs = set(range(total_fs))  # initially all FS are available


class Lightpath:
    def __init__(self, source: int, dest: int, transponder_mode: Dict, fs_allocation: Set[int],
                 capacity: float, used_capacity: float = 0.0):
        self.source = source
        self.dest = dest
        self.transponder_mode = transponder_mode  # dict with 'capacity', 'fs_required', 'max_spans'
        self.fs_allocation = fs_allocation  # set of frequency slots used
        self.capacity = capacity  # in Gb/s
        self.used_capacity = used_capacity  # in Gb/s
        self.demands = []  # list of demands carried by this lightpath

    def can_accommodate(self, demand_capacity: float) -> bool:
        return (self.capacity - self.used_capacity) >= demand_capacity

    def add_demand(self, demand_capacity: float):
        self.used_capacity += demand_capacity
        self.demands.append(demand_capacity)


class EthernetDemand:
    def __init__(self, demand_id: int, source: int, dest: int, traffic_class: str,
                 timeslots_needed: int, holding_time: float,arrival_time: float):
        self.demand_id = demand_id
        self.source = source
        self.dest = dest
        self.traffic_class = traffic_class  # '10GE', '100GE', or '400GE'
        self.timeslots_needed = timeslots_needed
        self.holding_time = holding_time
        self.blocked = False
        self.arrival_time = arrival_time
        self.path = None  # will store the path if not blocked


class MultilayerEON:
    def __init__(self):
        self.nodes = {}  # node_id: NetworkNode
        self.fiber_links = []  # list of FiberLink objects
        self.lightpaths = []  # list of Lightpath objects
        self.demands = []  # list of EthernetDemand objects
        self.transponder_modes = self._initialize_transponder_modes()
        self.traffic_classes = {
            '10GE': {'timeslots': 1, 'arrival_rate': 1.0, 'holding_time': 1.0},
            '100GE': {'timeslots': 10, 'arrival_rate': 0.5, 'holding_time': 1.0},
            '400GE': {'timeslots': 40, 'arrival_rate': 0.2, 'holding_time': 1.0}
        }

    def _initialize_transponder_modes(self) -> List[Dict]:
        """Initialize transponder operational modes as per Table V in the paper"""
        modes = [
            {'capacity': 200, 'modem_bitrate': 200, 'modem_baudrate': 95,
             'fs_required': 19, 'max_spans': 125},
            {'capacity': 300, 'modem_bitrate': 300, 'modem_baudrate': 95,
             'fs_required': 19, 'max_spans': 88},
            {'capacity': 400, 'modem_bitrate': 400, 'modem_baudrate': 95,
             'fs_required': 19, 'max_spans': 54},
            {'capacity': 500, 'modem_bitrate': 500, 'modem_baudrate': 95,
             'fs_required': 19, 'max_spans': 35},
            {'capacity': 600, 'modem_bitrate': 600, 'modem_baudrate': 95,
             'fs_required': 19, 'max_spans': 18},
            {'capacity': 700, 'modem_bitrate': 700, 'modem_baudrate': 95,
             'fs_required': 19, 'max_spans': 9},
            {'capacity': 800, 'modem_bitrate': 800, 'modem_baudrate': 95,
             'fs_required': 19, 'max_spans': 4},
            {'capacity': 100, 'modem_bitrate': 100, 'modem_baudrate': 56,
             'fs_required': 12, 'max_spans': 130},
            {'capacity': 200, 'modem_bitrate': 200, 'modem_baudrate': 56,
             'fs_required': 12, 'max_spans': 61},
            {'capacity': 300, 'modem_bitrate': 300, 'modem_baudrate': 56,
             'fs_required': 12, 'max_spans': 34},
            {'capacity': 400, 'modem_bitrate': 400, 'modem_baudrate': 56,
             'fs_required': 12, 'max_spans': 10},
            {'capacity': 100, 'modem_bitrate': 100, 'modem_baudrate': 35,
             'fs_required': 8, 'max_spans': 75},
            {'capacity': 200, 'modem_bitrate': 200, 'modem_baudrate': 35,
             'fs_required': 8, 'max_spans': 16}
        ]
        return modes

    def add_node(self, node_id: int, otn_switching_capacity: float = 24000.0):
        """Add a network node with OTN switching capability"""
        self.nodes[node_id] = NetworkNode(node_id, otn_switching_capacity)

    def add_fiber_link(self, source: int, dest: int, length: float):
        """Add a bidirectional fiber link between two nodes"""
        self.fiber_links.append(FiberLink(source, dest, length))
        self.fiber_links.append(FiberLink(dest, source, length))  # bidirectional

    def generate_demands(self, n: int, random_seed: int = None,
                         arrival_rate: float = 1.0, holding_rate: float = 1.0) -> List[EthernetDemand]:
        """
        Generate n demands with:
        - Arrival times following Poisson process (parameter λ=arrival_rate)
        - Holding times following Exponential distribution (parameter μ=holding_rate)
        - Controlled randomness using random_seed

        Args:
            n: Number of demands to generate
            random_seed: Seed for random number generation (for reproducibility)
            arrival_rate: λ parameter for Poisson arrival process
            holding_rate: μ parameter for Exponential holding times

        Returns:
            List of generated EthernetDemand objects with sequential demand_ids
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # Generate inter-arrival times (Exponential with rate λ)
        inter_arrivals = np.random.exponential(scale=1 / arrival_rate, size=n)

        # Calculate arrival times (cumulative sum of inter-arrivals)
        arrival_times = np.cumsum(inter_arrivals)

        # Generate holding times (Exponential with rate μ)
        holding_times = np.random.exponential(scale=1 / holding_rate, size=n)

        demands = []
        nodes = list(self.nodes.keys())

        for i in range(n):
            # Randomly select source and destination
            source, dest = random.sample(nodes, 2)

            # Randomly select traffic class weighted by relative rates
            traffic_class = random.choices(
                list(self.traffic_classes.keys()),
                weights=[self.traffic_classes[tc]['arrival_rate'] for tc in self.traffic_classes]
            )[0]

            timeslots_needed = self.traffic_classes[traffic_class]['timeslots']

            demands.append(EthernetDemand(
                demand_id=i,
                source=source,
                dest=dest,
                traffic_class=traffic_class,
                timeslots_needed=timeslots_needed,
                holding_time=holding_times[i],
                arrival_time=arrival_times[i]  # Adding arrival_time to the demand object
            ))

        return demands

    def find_k_shortest_paths(self, source: int, dest: int, k: int = 3) -> List[List[int]]:
        """Find K shortest paths between source and destination using Yen's algorithm"""

        # First, find the shortest path using Dijkstra's algorithm
        def dijkstra_shortest_path(adj: Dict[int, List[Tuple[int, float]]],
                                   source: int, dest: int,
                                   blocked_edges: Set[Tuple[int, int]] = None) -> Optional[List[int]]:
            """Dijkstra's algorithm with optional edge blocking"""
            heap = [(0, source, [])]  # (distance, node, path)
            visited = set()
            blocked_edges = blocked_edges or set()

            while heap:
                dist, node, path = heapq.heappop(heap)
                if node in visited:
                    continue
                visited.add(node)

                new_path = path + [node]

                if node == dest:
                    return new_path

                for neighbor, length in adj[node]:
                    if (node, neighbor) not in blocked_edges:
                        heapq.heappush(heap, (dist + length, neighbor, new_path))

            return None

        # Build adjacency list with distances
        adj = defaultdict(list)
        for link in self.fiber_links:
            adj[link.source].append((link.dest, link.length))

        # Get the shortest path
        shortest_path = dijkstra_shortest_path(adj, source, dest)
        if not shortest_path:
            return []

        # Initialize list of k shortest paths
        k_paths = [shortest_path]
        candidates = []

        for _ in range(1, k):
            # The previous path to work on
            prev_path = k_paths[-1]

            # Iterate through each node in the previous path (except the last one)
            for i in range(len(prev_path) - 1):
                # The spur node is the i-th node in the previous path
                spur_node = prev_path[i]

                # The root path is from source to spur node
                root_path = prev_path[:i + 1]

                # Initialize blocked edges
                blocked_edges = set()

                # Block all edges that are part of the previous paths that share the same root path
                for path in k_paths:
                    if len(path) > i and root_path == path[:i + 1]:
                        u = path[i]
                        v = path[i + 1] if i + 1 < len(path) else None
                        if v:
                            blocked_edges.add((u, v))

                # Find the spur path from spur node to destination
                spur_path = dijkstra_shortest_path(adj, spur_node, dest, blocked_edges)

                if spur_path:
                    # Combine root path and spur path
                    total_path = root_path[:-1] + spur_path

                    # Check if this path is already in candidates
                    if total_path not in candidates:
                        # Calculate total distance
                        total_distance = 0
                        for j in range(len(total_path) - 1):
                            u = total_path[j]
                            v = total_path[j + 1]
                            for neighbor, length in adj[u]:
                                if neighbor == v:
                                    total_distance += length
                                    break

                        # Add to candidates
                        heapq.heappush(candidates, (total_distance, total_path))

            # If there are no candidates, we've found all possible paths
            if not candidates:
                break

            # Add the shortest candidate path to k_paths
            _, new_path = heapq.heappop(candidates)
            k_paths.append(new_path)

        return k_paths[:k]

    def find_parallel_els(self, source: int, dest: int, damand: EthernetDemand) -> List[Lightpath]:
        """Find existing lightpaths between source and dest with available capacity"""
        parallel_els = []
        for lp in self.lightpaths:
            # zzg 修改为需要大于等于请求的需求才可以
            if lp.source == source and lp.dest == dest and lp.can_accommodate(damand.timeslots_needed * 10):
                parallel_els.append(lp)
        return parallel_els

    def find_parallel_eels(self, source: int, dest: int, damand: EthernetDemand) -> List[Lightpath]:
        """Find extendable existing lightpaths between source and dest"""
        # Simplified implementation - in practice would check if spectrum can be extended
        parallel_eels = []
        for lp in self.lightpaths:
            if lp.source == source and lp.dest == dest and not lp.can_accommodate(damand.timeslots_needed * 10):
                # Check if we can extend spectrum allocation
                # For now, just return all lightpaths that are at capacity
                parallel_eels.append(lp)
        return parallel_eels

    def check_pl(self, source: int, dest: int, demand: EthernetDemand) -> Optional[Lightpath]:
        """Check if a potential lightpath can be established between source and dest"""
        # Find the shortest path in G0 (physical layer) between source and dest
        path = self.find_shortest_path_in_g0(source, dest)
        if not path:
            return None

        # Check if the path length is within reach of any transponder mode
        path_length = sum(link.length for link in self.get_fiber_links_for_path(path))

        # Select appropriate transponder mode
        selected_mode = None
        for mode in sorted(self.transponder_modes, key=lambda x: x['fs_required']):
            if path_length <= mode['max_spans']:
                selected_mode = mode
                break

        if not selected_mode:
            return None

        # Check for contiguous FS availability along all fiber links in the path
        required_fs = selected_mode['fs_required']
        available_fs_sets = []

        for link in self.get_fiber_links_for_path(path):
            available_fs = link.available_fs
            # Find contiguous blocks of required_fs
            contiguous_blocks = self.find_contiguous_blocks(available_fs, required_fs)
            if not contiguous_blocks:
                return None
            available_fs_sets.append(set(contiguous_blocks[0]))  # take first available block

        # Find common contiguous FS across all links
        common_fs = set.intersection(*available_fs_sets)
        if not common_fs:
            return None

        # Create the potential lightpath
        fs_allocation = set(sorted(common_fs)[:required_fs])  # take first required_fs slots
        return Lightpath(source, dest, selected_mode, fs_allocation, selected_mode['capacity'])

    def find_shortest_path_in_g0(self, source: int, dest: int) -> Optional[List[int]]:
        """Find shortest path in physical layer (G0) using Dijkstra's algorithm"""
        # Build adjacency list
        adj = defaultdict(list)
        for link in self.fiber_links:
            adj[link.source].append((link.dest, link.length))

        # Dijkstra's algorithm
        heap = [(0, source, [])]  # (distance, node, path)
        visited = set()

        while heap:
            dist, node, path = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            new_path = path + [node]

            if node == dest:
                return new_path

            for neighbor, length in adj[node]:
                if neighbor not in visited:
                    heapq.heappush(heap, (dist + length, neighbor, new_path))

        return None

    def get_fiber_links_for_path(self, path: List[int]) -> List[FiberLink]:
        """Get fiber links corresponding to a path in G0"""
        links = []
        for i in range(len(path) - 1):
            source = path[i]
            dest = path[i + 1]
            for link in self.fiber_links:
                if link.source == source and link.dest == dest:
                    links.append(link)
                    break
        return links

    def find_contiguous_blocks(self, available_fs: Set[int], required_size: int) -> List[List[int]]:
        """Find all contiguous blocks of required_size in available_fs"""
        if not available_fs:
            return []

        sorted_fs = sorted(available_fs)
        blocks = []
        current_block = [sorted_fs[0]]

        for fs in sorted_fs[1:]:
            if fs == current_block[-1] + 1:
                current_block.append(fs)
            else:
                if len(current_block) >= required_size:
                    blocks.append(current_block)
                current_block = [fs]

        if len(current_block) >= required_size:
            blocks.append(current_block)

        return blocks

    def build_cag(self, demand: EthernetDemand, k: int = 3) -> Dict[Tuple[int, int], Lightpath]:
        """Build the Collapsed Auxiliary Graph for a given demand"""
        cag_edges = {}  # (source, dest): Lightpath

        # Find K shortest paths in G0
        paths = self.find_k_shortest_paths(demand.source, demand.dest, k)

        for path in paths:
            for i in range(len(path) - 1):
                source = path[i]
                dest = path[i + 1]

                if (source, dest) in cag_edges:
                    continue  # already added

                # Check for existing lightpaths (ELs)
                els = self.find_parallel_els(source, dest, demand)
                if els:
                    # Select the best EL (simplified: first one with enough capacity)
                    # zzg 修改为选择剩余容量最小的el作为best_el, p8,c2,p3
                    best_el = els[0]
                    for el in els:
                        if el.capacity - el.used_capacity < best_el.capacity - best_el.used_capacity:
                            best_el=el
                    cag_edges[(source, dest)] = best_el
                    continue

                # Check for extendable existing lightpaths (EELs)
                eels = self.find_parallel_eels(source, dest, demand)
                if eels:
                    # Select the best EEL (simplified: first one)
                    # zzg 修改为选择ABP最小的作为best_eel, p8,c2,p3
                    best_eel = eels[0]
                    cag_edges[(source, dest)] = best_eel
                    continue

                # Check for potential lightpaths (PLs)
                pl = self.check_pl(source, dest, demand)
                if pl:
                    cag_edges[(source, dest)] = pl

        return cag_edges

    def calculate_edge_weight(self, edge_type: str, lightpath: Lightpath, policy: str) -> float:
        """Calculate edge weight based on traffic engineering policy"""
        # Edge types: 'EL', 'EEL', 'PL'
        # Policies: 'MinEn', 'MaxMux', 'MaxSE', 'MinPB'

        # Get parameters from Table IV in the paper
        policy_params = {
            'MinEn': {'c0': 100000, 'c0_new': 10000, 'c_new': 1000, 'cf': 10, 'c_prime': 0, 'cu': 0, 'c_old': 1000},
            'MaxMux': {'c0': 0, 'c0_new': 1, 'c_new': 100, 'cf': 0, 'c_prime': 0, 'cu': 0, 'c_old': 0},
            'MaxSE': {'c0': 0, 'c0_new': 0, 'c_new': 1, 'cf': 0, 'c_prime': 0, 'cu': 1, 'c_old': 1},
            'MinPB': {'c0': 0.000001, 'c0_new': 0, 'c_new': 1000, 'cf': 0, 'c_prime': 0.001, 'cu': 0, 'c_old': 1000}
        }

        params = policy_params[policy]

        if edge_type == 'PL':
            # Calculate components for new lightpath
            h = lightpath.transponder_mode['max_spans']  # simplified as optical distance
            delta_f = 0  # simplified - would calculate actual fragmentation change
            u = lightpath.transponder_mode['modem_bitrate']

            weight = (params['c0'] +
                      params['c0_new'] +
                      params['c_new'] * h +
                      params['cf'] * delta_f +
                      params['c_prime'] * (h ** 2) +
                      params['cu'] * (10 ** (-u)))
        else:  # EL or EEL
            h = lightpath.transponder_mode['max_spans']  # simplified
            weight = params['c0'] + params['c_old'] * h

        return weight

    def label_setting_algorithm(self, cag_edges: Dict[Tuple[int, int], Lightpath],
                                source: int, dest: int, policy: str, max_hops: int = 5) -> Optional[List[int]]:
        """Label setting algorithm for constrained shortest path problem"""
        # Label format: (current_node, total_distance, used_fs_links, num_hops, path)

        # Priority queue for labels
        heap = []
        initial_label = (source, 0, set(), 0, [source])
        heapq.heappush(heap, (0, initial_label))  # priority is total_distance

        best_solution = None
        best_distance = float('inf')

        while heap:
            priority, label = heapq.heappop(heap)
            current_node, distance, used_fs_links, num_hops, path = label

            if current_node == dest and distance < best_distance:
                best_solution = path
                best_distance = distance
                continue

            if num_hops >= max_hops:
                continue

            # Explore neighbors in CAG
            for (u, v), lightpath in cag_edges.items():
                if u != current_node:
                    continue

                # Calculate new label components
                new_distance = distance + self.calculate_edge_weight(
                    'PL' if lightpath not in self.lightpaths else 'EL' if lightpath.can_accommodate(1.0) else 'EEL',
                    lightpath,
                    policy
                )

                new_used_fs_links = used_fs_links.copy()
                if lightpath not in self.lightpaths or not lightpath.can_accommodate(1.0):
                    # For PLs and EELs, track used FS links
                    # Simplified: just track the lightpath's FS allocation
                    new_used_fs_links.update(lightpath.fs_allocation)

                new_num_hops = num_hops + 1
                new_path = path + [v]

                new_label = (v, new_distance, new_used_fs_links, new_num_hops, new_path)

                # Check if new label is dominated by existing labels
                # Simplified dominance check
                if new_distance < best_distance:
                    heapq.heappush(heap, (new_distance, new_label))

        return best_solution

    def multilayer_rsa(self, demand: EthernetDemand, policy: str = 'MinPB', k: int = 3, max_hops: int = 5) -> bool:
        """Main algorithm to handle a new demand"""
        # Step 1: Build CAG
        cag_edges = self.build_cag(demand, k)

        # Step 2: Run label setting algorithm
        path = self.label_setting_algorithm(cag_edges, demand.source, demand.dest, policy, max_hops)

        if not path:
            demand.blocked = True
            return False

        # Step 3: Allocate resources along the path
        for i in range(len(path) - 1):
            source = path[i]
            dest = path[i + 1]
            lightpath = cag_edges[(source, dest)]

            if lightpath in self.lightpaths:
                # Existing lightpath (EL or EEL)
                if not lightpath.can_accommodate(demand.timeslots_needed):
                    # Need to extend EEL
                    # Simplified: just add to used capacity
                    lightpath.add_demand(demand.timeslots_needed)
            else:
                # Potential lightpath (PL)
                # Add to network
                self.lightpaths.append(lightpath)
                lightpath.add_demand(demand.timeslots_needed)

                # Update fiber links with FS allocation
                # Find physical path for this lightpath
                physical_path = self.find_shortest_path_in_g0(source, dest)
                fiber_links = self.get_fiber_links_for_path(physical_path)

                for link in fiber_links:
                    link.used_fs.update(lightpath.fs_allocation)
                    link.available_fs.difference_update(lightpath.fs_allocation)

            # Update OTN switching capacity at intermediate nodes
            if i > 0:  # not the first node
                self.nodes[source].used_otn_switching += demand.timeslots_needed

        demand.path = path
        return True

    def simulate(self, num_demands: int = 1000, policy: str = 'MinPB', k: int = 3, max_hops: int = 5):
        """Run simulation with given parameters"""

        # Generate 1000 demands with seed=42, λ=1.0, μ=1.0
        self.demands = self.generate_demands(n=1000, random_seed=42,
                                           arrival_rate=1.0, holding_rate=1.0)
        for demand in self.demands:
            self.multilayer_rsa(demand, policy, k, max_hops)


    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        metrics = {
            'blocking_ratio': sum(1 for d in self.demands if d.blocked) / len(self.demands),
            'avg_spectrum_usage': self.calculate_avg_spectrum_usage(),
            'avg_hops_per_demand': self.calculate_avg_hops(),
            'avg_otn_switching': self.calculate_avg_otn_switching(),
            'avg_multi_demand_lightpath_ratio': self.calculate_multi_demand_lightpath_ratio(),
            'avg_lightpath_capacity_usage': self.calculate_avg_lightpath_capacity_usage()
        }
        return metrics

    def calculate_avg_spectrum_usage(self) -> float:
        """Calculate average spectrum usage percentage"""
        total_used = sum(len(link.used_fs) for link in self.fiber_links)
        total_available = sum(link.total_fs for link in self.fiber_links)
        return (total_used / total_available) * 100 if total_available > 0 else 0

    def calculate_avg_hops(self) -> float:
        """Calculate average number of hops per demand"""
        successful_demands = [d for d in self.demands if not d.blocked and d.path]
        if not successful_demands:
            return 0
        return sum(len(d.path) - 1 for d in successful_demands) / len(successful_demands)

    def calculate_avg_otn_switching(self) -> float:
        """Calculate average OTN switching usage"""
        total_switching = sum(node.used_otn_switching for node in self.nodes.values())
        return total_switching / len(self.nodes) if self.nodes else 0

    def calculate_multi_demand_lightpath_ratio(self) -> float:
        """Calculate ratio of lightpaths serving multiple demands"""
        if not self.lightpaths:
            return 0
        multi_demand_lps = sum(1 for lp in self.lightpaths if len(lp.demands) > 1)
        return (multi_demand_lps / len(self.lightpaths)) * 100

    def calculate_avg_lightpath_capacity_usage(self) -> float:
        """Calculate average lightpath capacity utilization"""
        if not self.lightpaths:
            return 0
        total_utilization = sum(lp.used_capacity / lp.capacity for lp in self.lightpaths)
        return (total_utilization / len(self.lightpaths)) * 100


def create_sample_network(filename: str = "network_topology.txt") -> MultilayerEON:
    """Create and return a MultilayerEON instance from a text file with format:
    First line: n m  # number of nodes and edges
    Following m lines: s t d  # source, target, distance (km)
    """
    network = MultilayerEON()

    try:
        with open(filename, 'r') as f:
            # Read first line
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError("Empty file")

            # Parse n and m
            try:
                n, m = map(int, first_line.split())
            except ValueError:
                raise ValueError("First line should contain two integers")

            # Create nodes
            for node_id in range(1, n + 1):
                network.add_node(node_id)

            # Read edges
            edges_added = 0
            for line_num, line in enumerate(f, start=2):  # line numbers start at 2
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                try:
                    s, t, d = map(int, line.split())
                    if s < 1 or s > n or t < 1 or t > n:
                        raise ValueError(f"Node ID out of range in line {line_num}")
                    if d <= 0:
                        raise ValueError(f"Distance must be positive in line {line_num}")

                    network.add_fiber_link(s, t, d)
                    edges_added += 1
                except ValueError as e:
                    raise ValueError(f"Error in line {line_num}: {str(e)}")

            # Verify we read the correct number of edges
            if edges_added != m:
                raise ValueError(f"Expected {m} edges but found {edges_added}")

    except IOError as e:
        raise IOError(f"Error reading file {filename}: {str(e)}")

    return network


def run_performance_evaluation():
    """Run performance evaluation with different policies"""
    policies = ['MinEn', 'MaxMux', 'MaxSE', 'MinPB']
    results = {policy: {} for policy in policies}
    topology_file = "./topology/topology_example.txt"
    for policy in policies:
        print(f"Running simulation with {policy} policy...")
        network = create_sample_network(topology_file)
        network.simulate(num_demands=1000, policy=policy)
        metrics = network.get_performance_metrics()
        results[policy] = metrics

    return results


def plot_results(results: Dict):
    """Plot the performance metrics"""
    import matplotlib.pyplot as plt

    metrics = [
        'blocking_ratio',
        'avg_spectrum_usage',
        'avg_hops_per_demand',
        'avg_otn_switching',
        'avg_multi_demand_lightpath_ratio',
        'avg_lightpath_capacity_usage'
    ]

    titles = [
        'Blocking Ratio Comparison',
        'Average Spectrum Usage (%)',
        'Average Hops per Demand',
        'Average OTN Switching (Gb/s)',
        'Multi-Demand Lightpaths (%)',
        'Average Lightpath Capacity Usage (%)'
    ]

    ylabels = [
        'Blocking Ratio',
        'Spectrum Usage (%)',
        'Number of Hops',
        'OTN Switching (Gb/s)',
        'Percentage',
        'Capacity Usage (%)'
    ]

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[policy][metric] for policy in results]
        ax.bar(results.keys(), values)
        ax.set_title(titles[i])
        ax.set_ylabel(ylabels[i])
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()


# Main execution
if __name__ == "__main__":
    # Run performance evaluation
    results = run_performance_evaluation()

    # Print results
    print("\nPerformance Metrics Comparison:")
    for policy, metrics in results.items():
        print(f"\nPolicy: {policy}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")

    # Plot results
    plot_results(results)