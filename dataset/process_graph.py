import argparse
import os

def process_graph(input_filepath: str):
    """
    Process a graph file by removing self-loops and reindexing nodes.
    
    Args:
        input_filepath (str): Path to the input graph file
    """
    # Generate output path in the same folder with "_processed" suffix
    input_dir = os.path.dirname(input_filepath)
    input_filename = os.path.basename(input_filepath)
    name, ext = os.path.splitext(input_filename)
    output_filepath = os.path.join(input_dir, f"{name}_processed{ext}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First, count total lines and find maximum vertex ID for progress tracking
    print("Analyzing input file...")
    max_vertex_id = -1
    total_lines = 0
    
    # First pass: count total lines for progress bar
    with open(input_filepath, "r") as f:
        total_lines = sum(1 for line in f)
    
    # Second pass: analyze vertex IDs with progress bar
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
    
    print(f"Total lines in input file: {total_lines:,}")
    print(f"Maximum vertex ID (number of vertices): {max_vertex_id + 1:,}")

    # Open input and output files
    input_file = open(input_filepath, "r")
    output_file = open(output_filepath, "w")

    # Dictionaries for node mapping
    node_mapping = {}  # Maps original node IDs to new sequential IDs
    reverse_mapping = {}  # Maps new sequential IDs back to original node IDs

    # Statistics tracking
    max_index_before = -1
    max_index_after = -1
    num_lines_before = 0
    num_lines_after = 0
    processed_lines = 0

    print("Starting graph processing...")
    
    # Process each line in the input file
    for index, row in enumerate(input_file.readlines()):
        processed_lines += 1
        
        # Show progress every 100,0000 lines or every 1% of total lines
        if processed_lines % max(1000000, total_lines // 100) == 0:
            progress = (processed_lines / total_lines) * 100
            print(f"Progress: {processed_lines:,}/{total_lines:,} lines ({progress:.1f}%) - "
                  f"Processed edges: {num_lines_after:,}, Unique nodes: {len(node_mapping):,}")
        
        # Skip empty lines and header lines (starting with # or %)
        if not row.strip() or row.strip().startswith('#') or row.strip().startswith('%'):
            continue
        
        # Split by both tab and space
        parts = row.strip().replace('\t', ' ').split()
        if len(parts) < 2:
            continue
            
        # Handle both unweighted (src dest) and weighted (src dest weight) formats
        source, target = parts[0], parts[1]
        # Check if this is a weighted graph (has 3 parts)
        is_weighted = len(parts) >= 3
        weight = parts[2] if is_weighted else None
        
        # Skip self-loops
        if source == target:
            continue
            
        num_lines_before += 1
        
        # Map source node to new ID if not already mapped
        if source not in node_mapping:
            reverse_mapping[len(node_mapping)] = source
            node_mapping[source] = len(node_mapping)

        # Map target node to new ID if not already mapped
        if target not in node_mapping:
            reverse_mapping[len(node_mapping)] = target
            node_mapping[target] = len(node_mapping)

        # Write the reindexed edge to output file
        if is_weighted:
            output_file.write(f"{node_mapping[source]} {node_mapping[target]} {weight}\n")
        else:
            output_file.write(f"{node_mapping[source]} {node_mapping[target]}\n")

        # Update statistics
        max_index_before = max(max_index_before, int(source), int(target))
        max_index_after = max(max_index_after, node_mapping[source], node_mapping[target])
        num_lines_after += 1

    # Close files
    input_file.close()
    output_file.close()

    # Print final processing statistics
    print(f"\nProcessing completed!")
    print(f"Original max index: {max_index_before:,}, Original edge count: {num_lines_before:,}")
    print(f"Processed max index: {max_index_after:,}, Processed edge count: {num_lines_after:,}")
    print(f"Unique nodes: {len(node_mapping):,}")
    print(f"Output saved to: {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Process a graph file by removing self-loops and reindexing nodes.')
    parser.add_argument('--graph', required=True, help='Path to the input graph file')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.graph):
        print(f"Error: Input file '{args.graph}' does not exist")
        return
    
    process_graph(args.graph)

if __name__ == "__main__":
    main()