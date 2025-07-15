import argparse
import os
import gzip

def convert_graph(input_file: str):
    """
    Convert a .gr graph file to .txt format.
    
    Args:
        input_file (str): Path to the input .gr file (can be .gr or .gr.gz)
    """
    # Generate output path in the same folder as input with .txt extension
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    name, ext = os.path.splitext(input_filename)
    # Handle .gz extension
    if ext == '.gz':
        name, _ = os.path.splitext(name)  # Remove the .gr part
    output_file = os.path.join(input_dir, f"{name}.txt")
    
    # Check if file is gzipped
    is_gzipped = input_file.endswith('.gz')
    
    if is_gzipped:
        with gzip.open(input_file, "rt") as fin, open(output_file, "w") as fout:
            for line in fin:
                if line.startswith("a"):
                    parts = line.strip().split()
                    fout.write(f"{parts[1]} {parts[2]} {parts[3]}\n")
    else:
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            for line in fin:
                if line.startswith("a"):
                    parts = line.strip().split()
                    fout.write(f"{parts[1]} {parts[2]} {parts[3]}\n")
    
    print(f"Converted {input_file} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert a .gr graph file to .txt format.')
    parser.add_argument('--graph', required=True, help='Path to the input .gr file')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.graph):
        print(f"Error: Input file '{args.graph}' does not exist")
        return
    
    convert_graph(args.graph)

if __name__ == "__main__":
    main() 