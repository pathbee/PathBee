import argparse
import subprocess
import os

# Helper to run a command and print it
def run_cmd(cmd, **kwargs):
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, **kwargs)
        if result.returncode != 0:
            print(f"Command failed: {cmd}")
            exit(result.returncode)
    except Exception as e:
        print(f"Command failed: {cmd}")
        print(f"Error: {e}")
        exit(1)


def main():
    parser = argparse.ArgumentParser(description='End-to-end pipeline for PathBee')
    parser.add_argument('--graph-path', type=str, nargs='+', required=True, help='Path(s) to the graph file(s)')

    args = parser.parse_args()

    run_cmd("g++ pathbee/algorithms/2_hop_labeling.cpp -o 2_hop_labeling")

    for graph_path in args.graph_path:
        graph_name = os.path.basename(graph_path)
        centrality_folder = os.path.join("datasets/centralities", graph_name)
        os.makedirs(centrality_folder, exist_ok=True)

        # 1. Calculate centrality (dc)
        # run_cmd(f"python launch.py cen --graph-path {graph_path} --save-dir {centrality_folder} --centrality dc")

        # # # 2. Inference
        # run_cmd(f"python launch.py infer --graph-path {graph_path} --save-dir {centrality_folder} --model-path models/model.pt --adj-size 4500000")

        # 3. Construct index (using dc and gnn centrality)
        # run_cmd(f"python launch.py index --graph-path {graph_path} --centrality-path {centrality_folder}/dc_ranking.txt --index-path bin/{graph_name.replace('.txt', '_dc.bin')}")
        # run_cmd(f"python launch.py index --graph-path {graph_path} --centrality-path {centrality_folder}/gnn_ranking.txt --index-path bin/{graph_name.replace('.txt', '_gnn.bin')}")

        # 4. Query (example: plot/query time distribution) 
        result_dir = f"results/{graph_name}/"

        # random sampling
        run_cmd(f"python launch.py query --index-path bin/{graph_name.replace('.txt', '_dc.bin')} bin/{graph_name.replace('.txt', '_gnn.bin')} --graph-path {graph_path} --num-queries 100000 --result-dir {result_dir}")
        # # stratified sampling
        run_cmd(f"python launch.py query --index-path bin/{graph_name.replace('.txt', '_dc.bin')} bin/{graph_name.replace('.txt', '_gnn.bin')} --graph-path {graph_path} --num-queries 100000 --result-dir {result_dir} --stratified")

if __name__ == "__main__":
    main()
