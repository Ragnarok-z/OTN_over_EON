import heapq
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from NetworkNode import NetworkNode
from FiberLink import FiberLink
from Lightpath import Lightpath
from EthernetDemand import EthernetDemand
from MultilayerEON import MultilayerEON

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