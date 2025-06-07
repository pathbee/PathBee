import subprocess
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

def get_num_vertices(graph_file):
    max_vertex = 0
    with open(graph_file, 'r') as f:
        for line in f:
            v1, v2 = map(int, line.strip().split())
            max_vertex = max(max_vertex, v1, v2)
    return max_vertex + 1  # +1 because vertices are 0-based

def run_queries(index_path, query_pairs):
    process = subprocess.Popen(['./query_distance', index_path],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    process.stdout.readline()  # Skip "Index loaded successfully..."
    process.stdout.readline()  # Skip "Enter queries..."

    query_times = []
    for start, end in tqdm(query_pairs, desc=f"Querying {index_path}"):
        start_time = time.time()
        process.stdin.write(f"{start} {end}\n")
        process.stdin.flush()
        process.stdout.readline()
        end_time = time.time()
        query_times.append(end_time - start_time)

    process.stdin.write("-1\n")
    process.stdin.flush()
    process.wait()
    return np.array(query_times) * 1_000_000  # microseconds

def main():
    parser = argparse.ArgumentParser(description='Measure query time distribution for random queries')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to the graph file')
    parser.add_argument('--index_path', type=str, nargs='+', required=True, help='Path(s) to the index file(s)')
    parser.add_argument('--num_queries', type=int, default=100000, help='Number of random queries to perform')
    parser.add_argument('--output', type=str, default='query_time_distribution.png', help='Output plot filename')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    num_vertices = get_num_vertices(args.graph_path)
    print(f"Number of vertices in graph: {num_vertices}")

    # Set random seed and generate all query pairs once
    random.seed(args.seed)
    np.random.seed(args.seed)
    query_pairs = [(random.randint(0, num_vertices - 1), random.randint(0, num_vertices - 1))
                   for _ in range(args.num_queries)]

    all_query_times = []
    labels = []
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, index_path in enumerate(args.index_path):
        print(f"\nRunning queries for {index_path}...")
        query_times = run_queries(index_path, query_pairs)
        all_query_times.append(query_times)
        labels.append(index_path)

        # Print stats for each method
        mean_time = np.mean(query_times)
        median_time = np.median(query_times)
        p95_time = np.percentile(query_times, 95)
        p99_time = np.percentile(query_times, 99)
        p999_time = np.percentile(query_times, 99.9)
        min_time = np.min(query_times)
        max_time = np.max(query_times)
        print(f"\n[{index_path}] Query Time Statistics (microseconds):")
        print(f"Min: {min_time:.2f}")
        print(f"Max: {max_time:.2f}")
        print(f"Mean: {mean_time:.2f}")
        print(f"Median: {median_time:.2f}")
        print(f"95th percentile: {p95_time:.2f}")
        print(f"99th percentile: {p99_time:.2f}")
        print(f"99.9th percentile: {p999_time:.2f}")

    # Plot all methods
    plt.figure(figsize=(12, 6))
    for i, query_times in enumerate(all_query_times):
        plt.hist(query_times, bins=100, density=True, alpha=0.5, color=colors[i % len(colors)], label=labels[i])

    plt.title('Distribution of Query Times (Random Queries)', fontsize=16, weight='bold')
    plt.xlabel('Query Time (microseconds)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # X-axis from min to 99th percentile of all data
    all_times = np.concatenate(all_query_times)
    min_time = np.min(all_times)
    p99_time = np.percentile(all_times, 99)
    plt.xlim(min_time, p99_time)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"\nPlot saved as '{args.output}'")

if __name__ == "__main__":
    main()
