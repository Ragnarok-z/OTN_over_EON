import matplotlib.pyplot as plt
import json
import os

from Network import Network
from Simulator import Simulator
from Tool import get_next_exp_number

def run_experiments(topology_file, output_dir="results"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 获取下一个实验编号并创建子目录
    exp_number = get_next_exp_number(output_dir)
    output_dir = os.path.join(output_dir, f"exp_{exp_number}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")
    # Load network topology
    network = Network(topology_file)

    # Define traffic intensities (Erlang)
    traffic_intensities = range(300, 510, 40)

    # Define number of demand
    num_demands = 10000

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
    output_dir = "results"
    run_experiments(topology_file=topology_file, output_dir=output_dir)