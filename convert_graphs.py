#!/usr/bin/env python3
import os
import subprocess
import tempfile
import shutil
import gzip
import zipfile

def convert_tsv_file(input_path, output_path):
    """
    Convert the TSV file (3 columns) to the required format (2 columns: source dest)
    """
    print(f"Converting {input_path} to {output_path}")
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # Take first two columns as source and destination
                source = parts[0]
                dest = parts[1]
                outfile.write(f"{source} {dest}\n")
    
    print(f"Converted {input_path} successfully")

def convert_dimacs_tar(input_path, output_path):
    """
    Convert the DIMACS format from the compressed tar file to the required format
    """
    print(f"Converting {input_path} to {output_path}")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        # Extract the tar file
        subprocess.run(['tar', '-xf', input_path, '-C', tempdir], check=True)
        
        # Find the dimacs9-USA directory
        dimacs_dir = os.path.join(tempdir, 'dimacs9-USA')
        if not os.path.exists(dimacs_dir):
            raise FileNotFoundError(f"Expected directory {dimacs_dir} not found")
        
        # Read the graph file
        graph_file = os.path.join(dimacs_dir, 'out.dimacs9-USA')
        if not os.path.exists(graph_file):
            raise FileNotFoundError(f"Graph file {graph_file} not found")
        
        with open(graph_file, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                line = line.strip()
                # Skip comment lines and empty lines
                if line.startswith('%') or line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    source = parts[0]
                    dest = parts[1]
                    outfile.write(f"{source} {dest}\n")
    
    print(f"Converted {input_path} successfully")

def convert_gzip_file(input_path, output_path):
    """
    Convert the gzipped file to the required format (already in correct format, just decompress)
    """
    print(f"Converting {input_path} to {output_path}")
    
    with gzip.open(input_path, 'rt') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if line:  # Skip empty lines
                outfile.write(f"{line}\n")
    
    print(f"Converted {input_path} successfully")

def convert_edges_zip(input_path, output_path):
    """
    Convert the edges file from zip to the required format
    """
    print(f"Converting {input_path} to {output_path}")
    
    with zipfile.ZipFile(input_path, 'r') as zip_file:
        # Find the .edges file
        edges_file = None
        for name in zip_file.namelist():
            if name.endswith('.edges'):
                edges_file = name
                break
        
        if not edges_file:
            raise FileNotFoundError(f"No .edges file found in {input_path}")
        
        with zip_file.open(edges_file, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                line = line.decode('utf-8').strip()
                # Skip comment lines and empty lines
                if line.startswith('%') or line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    source = parts[0]
                    dest = parts[1]
                    outfile.write(f"{source} {dest}\n")
    
    print(f"Converted {input_path} successfully")

def convert_mtx_zip(input_path, output_path):
    """
    Convert the Matrix Market format from zip to the required format
    """
    print(f"Converting {input_path} to {output_path}")
    
    with zipfile.ZipFile(input_path, 'r') as zip_file:
        # Find the .mtx file
        mtx_file = None
        for name in zip_file.namelist():
            if name.endswith('.mtx'):
                mtx_file = name
                break
        
        if not mtx_file:
            raise FileNotFoundError(f"No .mtx file found in {input_path}")
        
        with zip_file.open(mtx_file, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                line = line.decode('utf-8').strip()
                # Skip comment lines and empty lines
                if line.startswith('%') or line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    # Matrix Market format: row col value (we ignore the value)
                    source = parts[0]
                    dest = parts[1]
                    outfile.write(f"{source} {dest}\n")
    
    print(f"Converted {input_path} successfully")

def main():
    # Create output directory
    output_dir = "dataset/converted"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the Twitter gzip file
    twitter_input = "dataset/rw/twitter-2010.txt.gz"
    twitter_output = os.path.join(output_dir, "twitter-2010.txt")
    
    if os.path.exists(twitter_input):
        convert_gzip_file(twitter_input, twitter_output)
    else:
        print(f"Warning: {twitter_input} not found")
    
    # Convert the Orkut edges zip file
    orkut_input = "dataset/rw/aff-orkut-user2groups.zip"
    orkut_output = os.path.join(output_dir, "aff-orkut-user2groups.txt")
    
    if os.path.exists(orkut_input):
        convert_edges_zip(orkut_input, orkut_output)
    else:
        print(f"Warning: {orkut_input} not found")
    
    # Convert the Twitter Matrix Market zip file
    twitter_mtx_input = "dataset/rw/soc-twitter-2010.zip"
    twitter_mtx_output = os.path.join(output_dir, "soc-twitter-2010.txt")
    
    if os.path.exists(twitter_mtx_input):
        convert_mtx_zip(twitter_mtx_input, twitter_mtx_output)
    else:
        print(f"Warning: {twitter_mtx_input} not found")
    
    print(f"\nConversion complete! Files saved in {output_dir}/")
    print("Converted files:")
    if os.path.exists(twitter_output):
        print(f"  - {twitter_output}")
    if os.path.exists(orkut_output):
        print(f"  - {orkut_output}")
    if os.path.exists(twitter_mtx_output):
        print(f"  - {twitter_mtx_output}")

if __name__ == "__main__":
    main() 