import os
import subprocess
import random
import time
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import csv

def get_num_vertices(graph_file):
    max_vertex = 0
    with open(graph_file, 'r') as f:
        for line in f:
            v1, v2 = map(int, line.strip().split())
            max_vertex = max(max_vertex, v1, v2)
    return max_vertex + 1  # +1 because vertices are 0-based

def run_queries(index_path, query_pairs, csv_output=None):
    process = subprocess.Popen(['./2_hop_labeling', 'distance', index_path],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    # For index item queries, open a second process
    process_index = subprocess.Popen(['./2_hop_labeling', 'index_items', index_path],
                                     stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True)
    process.stdout.readline()  # Skip "Index loaded successfully..."
    process.stdout.readline()  # Skip "Enter queries..."
    process_index.stdout.readline()  # Skip "Index loaded successfully..."
    process_index.stdout.readline()  # Skip "Enter vertex id..."

    results = []
    for start, end in tqdm(query_pairs, desc=f"Querying {index_path}"):
        # Query distance
        start_time = time.time()
        process.stdin.write(f"{start} {end}\n")
        process.stdin.flush()
        dist_line = process.stdout.readline()
        end_time = time.time()
        query_time = (end_time - start_time) * 1_000_000  # microseconds

        # Parse distance
        if "Distance between vertices" in dist_line:
            dist = int(dist_line.strip().split()[-1])
        else:
            dist = -1  # or INT_MAX

        # Query outIndex for v
        process_index.stdin.write(f"{start}\n")
        process_index.stdin.flush()
        v_index_line = process_index.stdout.readline()
        v_out_index_num = int(v_index_line.split('outIndex items = ')[-1])

        # Query inIndex for w
        process_index.stdin.write(f"{end}\n")
        process_index.stdin.flush()
        w_index_line = process_index.stdout.readline()
        w_in_index_num = int(w_index_line.split('inIndex items = ')[-1].split(',')[0])

        results.append([start, end, dist, query_time, v_out_index_num, w_in_index_num])

    process.stdin.write("-1\n")
    process.stdin.flush()
    process.wait()
    process_index.stdin.write("-1\n")
    process_index.stdin.flush()
    process_index.wait()

    if csv_output:
        # Print statistics before writing CSV
        query_times = np.array([row[3] for row in results])
        dists = np.array([row[2] for row in results])
        if len(query_times) > 0:
            print("Query Time Statistics (microseconds):")
            print(f"Min: {np.min(query_times):.2f}")
            print(f"Max: {np.max(query_times):.2f}")
            print(f"Mean: {np.mean(query_times):.2f}")
            print(f"Median: {np.median(query_times):.2f}")
            print(f"99th percentile: {np.percentile(query_times, 99):.2f}")
            print(f"99.9th percentile: {np.percentile(query_times, 99.9):.2f}")
            print(f"Number of queries: {len(query_times)}")
            print(f"Number of unreachable queries (dist == -1): {(dists == -1).sum()}")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_output), exist_ok=True)
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
    return results

def generate_query_csv(index_paths, graph_path, num_queries=100000, seed=42, csv_output=None):
    """
    Generate a CSV of query results (distance, query time, index sizes) for random queries.
    """
    num_vertices = get_num_vertices(graph_path)
    print(f"Number of vertices in graph: {num_vertices}")

    random.seed(seed)
    np.random.seed(seed)
    query_pairs = [(random.randint(0, num_vertices - 1), random.randint(0, num_vertices - 1))
                   for _ in range(num_queries)]

    for i, index_path in enumerate(index_paths):
        print(f"\nRunning queries for {index_path}...")
        run_queries(index_path, query_pairs, csv_output=csv_output)

def main():
    parser = argparse.ArgumentParser(description='Generate CSV of query results for random queries')
    parser.add_argument('--graph_path', type=str, required=True, help='Path to the graph file')
    parser.add_argument('--index_path', type=str, nargs='+', required=True, help='Path(s) to the index file(s)')
    parser.add_argument('--num_queries', type=int, default=100000, help='Number of random queries to perform')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--csv_output', type=str, default=None, help='Path to save the CSV file')
    args = parser.parse_args()

    generate_query_csv(args.index_path, args.graph_path, args.num_queries, args.seed, args.csv_output)

if __name__ == "__main__":
    main()
