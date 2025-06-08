import os
import subprocess
import random
import time
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import pandas as pd
import tempfile

def get_num_vertices(graph_file):
    max_vertex = 0
    with open(graph_file, 'r') as f:
        for line in f:
            v1, v2 = map(int, line.strip().split())
            max_vertex = max(max_vertex, v1, v2)
    return max_vertex + 1  # +1 because vertices are 0-based

def run_queries(index_paths, query_pairs, result_dir=None):
    """
    Run queries for multiple index files, saving results in result_dir with CSV named after each index file.
    Uses query_distance mode for efficiency.
    """
    results_all = []
    for index_path in index_paths:
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            csv_output = os.path.join(result_dir, os.path.splitext(os.path.basename(index_path))[0] + '.csv')
        else:
            csv_output = None
        with tempfile.TemporaryDirectory() as tmpdir:
            query_file = os.path.join(tmpdir, 'queries.txt')
            output_file = os.path.join(tmpdir, 'results.txt')
            # Write all query pairs to file
            with open(query_file, 'w') as f:
                for start, end in query_pairs:
                    f.write(f"{start} {end}\n")
            # Call query_distance
            subprocess.run(['./2_hop_labeling', 'distance', index_path, query_file, output_file], check=True)
            # Read results
            results = []
            with open(output_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 4:
                        continue
                    v, w, dist, query_time = parts
                    v = int(v)
                    w = int(w)
                    dist = int(dist)
                    query_time = float(query_time)
                    # For compatibility, set v_out_index_num and w_in_index_num to -1 (or add another batch mode if needed)
                    results.append([v, w, dist, query_time, -1, -1])
        if csv_output:
            query_times = np.array([row[3] for row in results])
            dists = np.array([row[2] for row in results])
            stats_output = os.path.join(result_dir, 'stats.txt')
            if len(query_times) > 0:
                with open(stats_output, 'a') as f:
                    f.write(f"Index: {index_path}\n")
                    f.write("Query Time Statistics (microseconds):\n")
                    f.write(f"Min: {np.min(query_times):.2f}\n")
                    f.write(f"Max: {np.max(query_times):.2f}\n")
                    f.write(f"Mean: {np.mean(query_times):.2f}\n")
                    f.write(f"Median: {np.median(query_times):.2f}\n")
                    f.write(f"99th percentile: {np.percentile(query_times, 99):.2f}\n")
                    f.write(f"99.9th percentile: {np.percentile(query_times, 99.9):.2f}\n")
                    f.write(f"Number of queries: {len(query_times)}\n")
                    f.write(f"Number of unreachable queries (dist == -1): {(dists == -1).sum()}\n")
            with open(csv_output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'v (vertex)',
                    'w (vertex)',
                    'dist (hops)',
                    'query_time (microseconds)',
                    'v_out_index_num (count)',
                    'w_in_index_num (count)'
                ])
                writer.writerows(results)
        results_all.append(results)
    return results_all

def generate_query_csv(index_paths, graph_path, num_queries=100000, seed=42, result_dir=None):
    """
    Generate CSVs of query results for random queries for multiple index files.
    """
    num_vertices = get_num_vertices(graph_path)
    print(f"Number of vertices in graph: {num_vertices}")

    random.seed(seed)
    np.random.seed(seed)
    query_pairs = [(random.randint(0, num_vertices - 1), random.randint(0, num_vertices - 1))
                   for _ in range(num_queries)]

    run_queries(index_paths, query_pairs, result_dir=result_dir)

def plot_query_time_distribution(result_dir):
    """
    Plot the query time distribution from all CSV files in result_dir.
    Args:
        result_dir (str): Directory containing the CSV files.
    """
    csv_paths = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith('.csv')]
    if not csv_paths:
        print(f"No CSV files found in {result_dir}")
        return
    method_names = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
    plt.figure(figsize=(8, 6))
    colors = ['#FDB863', '#B2ABD2', '#FC8D59', '#99D594', '#E66101', '#5E3C99']
    max_x = 0
    for i, (csv_path, method) in enumerate(zip(csv_paths, method_names)):
        df = pd.read_csv(csv_path)
        query_times = df['query_time (microseconds)'] / 1_000_000
        plt.hist(query_times, bins=1000, density=True, alpha=0.7, label=method, color=colors[i % len(colors)])
        max_x = max(max_x, np.percentile(query_times, 99.9))
        
    plt.xlim(0, max_x)
    plt.xlabel('Query Time (seconds)')
    plt.ylabel('Density')
    plt.title('Query-Time Distributions for Methods')
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(result_dir, 'query_time_distribution.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate CSV of query results for random queries')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to the graph file')
    parser.add_argument('--index_path', type=str, nargs='+', required=True, help='Path(s) to the index file(s)')
    parser.add_argument('--num_queries', type=int, default=100000, help='Number of random queries to perform')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--result_dir', type=str, default=None, help='Path to save the CSV files')
    args = parser.parse_args()

    generate_query_csv(args.index_path, args.graph_path, args.num_queries, args.seed, args.result_dir)

if __name__ == "__main__":
    main()
