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

def get_index_counts(index_path, vertices):
    """
    Get index item counts for a list of vertices using the index_items mode.
    More efficient version that processes all vertices in a single call.
    """
    counts = {}
    
    # Create a temporary file with all vertices
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        for v in vertices:
            f.write(f"{v}\n")
        temp_file = f.name
    
    try:
        # Run the index_items command once for all vertices
        result = subprocess.run(['./2_hop_labeling', 'index_items', index_path, temp_file], 
                              capture_output=True, text=True)
        
        # Parse the output to get counts
        for line in result.stdout.split('\n'):
            if line.startswith('Vertex '):
                parts = line.split(':')
                if len(parts) == 2:
                    vertex = int(parts[0].split()[1])
                    counts_str = parts[1].strip()
                    in_count = int(counts_str.split(',')[0].split('=')[1].strip())
                    out_count = int(counts_str.split(',')[1].split('=')[1].strip())
                    counts[vertex] = (in_count, out_count)
    finally:
        # Clean up the temporary file
        os.unlink(temp_file)
    
    return counts

def run_queries(index_paths, query_pairs, result_dir=None, stratified=False):
    """
    Run queries for multiple index files, saving results in result_dir with CSV named after each index file.
    Uses query_distance mode for efficiency.
    """
    results_all = []
    if stratified:
        result_dir = os.path.join(result_dir, 'stratified_sampling')
    else:
        result_dir = os.path.join(result_dir, 'random_sampling')
    for index_path in index_paths:
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            csv_output = os.path.join(result_dir, os.path.splitext(os.path.basename(index_path))[0] + '.csv')
        else:
            csv_output = None
            
        # Get unique vertices from query pairs
        unique_vertices = set()
        for v, w in query_pairs:
            unique_vertices.add(v)
            unique_vertices.add(w)
        
        # Get index counts for all vertices
        index_counts = get_index_counts(index_path, unique_vertices)
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
                    # Get actual index counts
                    v_out_count = index_counts[v][1] if v in index_counts else 0
                    w_in_count = index_counts[w][0] if w in index_counts else 0
                    results.append([v, w, dist, query_time, v_out_count, w_in_count])
        if csv_output:
            query_times = np.array([row[3] for row in results])
            dists = np.array([row[2] for row in results])
            stats_output = os.path.join(result_dir, 'stats.txt')
            if len(query_times) > 0:
                with open(stats_output, 'a') as f:
                    f.write("\n")
                    f.write(f"Index: {index_path}\n")
                    f.write("Query Time Statistics (microseconds):\n")
                    f.write(f"Min: {np.min(query_times):.2f}\n")
                    f.write(f"Max: {np.max(query_times):.2f}\n")
                    f.write(f"Mean: {np.mean(query_times):.2f}\n")
                    f.write(f"Median: {np.median(query_times):.2f}\n")
                    f.write(f"99th percentile: {np.percentile(query_times, 99):.2f}\n")
                    f.write(f"99.9th percentile: {np.percentile(query_times, 99.9):.2f}\n")
                    f.write(f"Number of queries: {len(query_times)}\n")
                    f.write(f"Number of unreachable queries (dist == INT_MAX): {(dists == 2147483647).sum()}\n")
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

def generate_query_csv(index_paths, graph_path, num_queries=100000, seed=42, result_dir=None, stratified=False):
    """
    Generate CSVs of query results for queries for multiple index files.
    Args:
        index_paths: List of paths to index files
        graph_path: Path to the graph file
        num_queries: Number of queries to generate
        seed: Random seed for reproducibility
        result_dir: Directory to save results
        stratified: If True, use stratified sampling based on index item counts. If False, use random sampling.
    """
    num_vertices = get_num_vertices(graph_path)
    print(f"Number of vertices in graph: {num_vertices}")

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    if not stratified:
        print("Using random sampling")
        query_pairs = [(random.randint(0, num_vertices - 1), random.randint(0, num_vertices - 1))
                      for _ in range(num_queries)]
    else:
        print("Using stratified sampling")
        # First get index counts for all vertices
        all_vertices = list(range(num_vertices))
        index_counts = get_index_counts(index_paths[0], all_vertices)  # Use first index file for binning
        
        # Create bins based on in/out index item counts
        in_counts = [counts[0] for counts in index_counts.values()]
        out_counts = [counts[1] for counts in index_counts.values()]
        
        # Create 10 equal-frequency bins (each containing 1% of vertices)
        in_bins = np.percentile(in_counts, np.linspace(0, 100, 11))
        out_bins = np.percentile(out_counts, np.linspace(0, 100, 11))
        
        print(f"\nIndex item count ranges for top 1% vertices:")
        print(f"In-count: {in_bins[-2]:.0f} to {in_bins[-1]:.0f}")
        print(f"Out-count: {out_bins[-2]:.0f} to {out_bins[-1]:.0f}")
        
        # Assign vertices to bins
        in_bin_assignments = np.digitize(in_counts, in_bins[:-1])
        out_bin_assignments = np.digitize(out_counts, out_bins[:-1])
        
        # Group vertices by their bin assignments
        in_bin_vertices = {i: [] for i in range(1, 11)}
        out_bin_vertices = {i: [] for i in range(1, 11)}
        
        for v, (in_bin, out_bin) in enumerate(zip(in_bin_assignments, out_bin_assignments)):
            in_bin_vertices[in_bin].append(v)
            out_bin_vertices[out_bin].append(v)
        
        # Generate query pairs by matching vertices from bins with maximum index item counts
        query_pairs = []
        queries_per_bin = num_queries
        
        for bin_idx in range(10, 11):
            in_vertices = in_bin_vertices[bin_idx]
            out_vertices = out_bin_vertices[bin_idx]
            
            if not in_vertices or not out_vertices:
                continue
                
            print(f"\nSelected vertices from bin {bin_idx} (top 1%)")
            print(f"Number of in-vertices: {len(in_vertices)}")
            print(f"Number of out-vertices: {len(out_vertices)}")
            
            for _ in range(queries_per_bin):
                v = random.choice(out_vertices)
                w = random.choice(in_vertices)
                query_pairs.append((v, w))
        
        # If we don't have enough pairs, fill the rest randomly
        while len(query_pairs) < num_queries:
            v = random.randint(0, num_vertices - 1)
            w = random.randint(0, num_vertices - 1)
            query_pairs.append((v, w))

    run_queries(index_paths, query_pairs, result_dir=result_dir, stratified=stratified)

def plot_query_time_distribution(result_dir, stratified=False):
    """
    Plot the query time distribution from all CSV files in result_dir.
    Args:
        result_dir (str): Directory containing the CSV files.
        stratified (bool): Whether to use stratified sampling directory.
    """
    if stratified:
        result_dir = os.path.join(result_dir, 'stratified_sampling')
    else:
        result_dir = os.path.join(result_dir, 'random_sampling')
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
    output_path = os.path.join(result_dir, 'query_time_distribution' + '.png')
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
    parser.add_argument('--stratified', type=bool, default=False, help='Whether to use stratified sampling')
    args = parser.parse_args()

    generate_query_csv(args.index_path, args.graph_path, args.num_queries, args.seed, args.result_dir, args.stratified)

if __name__ == "__main__":
    main()
