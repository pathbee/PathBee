import argparse
import os

def process_graph(input_filepath: str, output_filepath: str = None):
    """
    Process a graph file by removing self-loops and reindexing nodes.
    
    Args:
        input_filepath (str): Path to the input graph file
        output_filepath (str, optional): Path where the processed graph will be saved.
            If not provided, will overwrite the original file.
            If a directory is provided, will save with the same filename in that directory.
    """
    # If no output path is provided, overwrite the original file
    if output_filepath is None:
        output_filepath = input_filepath
    # If output path is a directory, use the input filename in that directory
    elif os.path.isdir(output_filepath):
        input_filename = os.path.basename(input_filepath)
        output_filepath = os.path.join(output_filepath, input_filename)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Process each line in the input file
    for index, row in enumerate(input_file.readlines()):
        # Skip empty lines and header lines (starting with #)
        if not row.strip() or row.strip().startswith('#'):
            continue
        
        # Split by both tab and space
        parts = row.strip().replace('\t', ' ').split()
        if len(parts) < 2:
            continue
            
        source, target = parts[0], parts[1]
        
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
        output_file.write(f"{node_mapping[source]} {node_mapping[target]}\n")

        # Update statistics
        max_index_before = max(max_index_before, int(source), int(target))
        max_index_after = max(max_index_after, node_mapping[source], node_mapping[target])
        num_lines_after += 1

    # Close files
    input_file.close()
    output_file.close()

    # Print processing statistics
    print(f"Original max index: {max_index_before}, Original edge count: {num_lines_before}")
    print(f"Processed max index: {max_index_after}, Processed edge count: {num_lines_after}")
    print(f"Output saved to: {output_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Process a graph file by removing self-loops and reindexing nodes.')
    parser.add_argument('--input', required=True, help='Path to the input graph file')
    parser.add_argument('--output', help='Path where the processed graph will be saved (optional, defaults to overwriting the input file)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    process_graph(args.input, args.output)

if __name__ == "__main__":
    main()