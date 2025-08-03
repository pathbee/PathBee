#!/usr/bin/env python3
import argparse
import os

def get_graph_stats(input_filepath: str):
    """
    Get basic statistics from a graph file: number of nodes and edges.
    
    Args:
        input_filepath (str): Path to the input graph file
    
    Returns:
        tuple: (num_nodes, num_edges)
    """
    max_vertex_id = -1
    num_edges = 0
    
    print(f"Analyzing graph file: {input_filepath}")
    
    # First, count total lines for progress tracking
    with open(input_filepath, "r") as f:
        total_lines = sum(1 for line in f)
    
    # Second pass: analyze with progress bar
    processed_lines = 0
    with open(input_filepath, "r") as f:
        for line in f:
            processed_lines += 1
            
            # Show progress every 100,000 lines or every 1% of total lines
            if processed_lines % max(100000, total_lines // 100) == 0:
                progress = (processed_lines / total_lines) * 100
                print(f"Analysis progress: {processed_lines:,}/{total_lines:,} lines ({progress:.1f}%)")
            
            # Skip empty lines and header lines (starting with # or %)
            if not line.strip() or line.strip().startswith('#') or line.strip().startswith('%'):
                continue
            
            # Split by both tab and space
            parts = line.strip().replace('\t', ' ').split()
            if len(parts) < 2:
                continue
                
            # Get source and target vertex IDs
            source, target = int(parts[0]), int(parts[1])
            max_vertex_id = max(max_vertex_id, source, target)
            num_edges += 1
    
    num_nodes = max_vertex_id + 1  # Vertex IDs are typically 0-indexed
    
    return num_nodes, num_edges

def main():
    parser = argparse.ArgumentParser(description='Get basic statistics from a graph file.')
    parser.add_argument('--graph', required=True, help='Path to the input graph file')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.graph):
        print(f"Error: Input file '{args.graph}' does not exist")
        return
    
    num_nodes, num_edges = get_graph_stats(args.graph)
    
    print(f"Number of nodes: {num_nodes:,}")
    print(f"Number of edges: {num_edges:,}")

if __name__ == "__main__":
    main()