# PathBee: Accelerating Shortest Path Querying via Graph Neural Networks

This repository hosts the source code and supplementary materials for our VLDB 2024 submission, "PathBee: Accelerating Shortest Path Querying via Graph Neural Networks". This work presents PathBee, an innovative framework leveraging Graph Neural Networks (GNNs) that significantly advances the current 2-hop labeling-based approaches.

## PathBee Workflow

Our workflow is comprised of four key stages:

1. **Training Data Preparation**: We generate synthetic scale-free graphs using the NetworkX library, and then annotate the data using an internal tool that calculates precise centrality values.
2. **GNN Model Training**: We use the prepared datasets to train our GNN model.
3. **GNN Model Inference**: This stage is dedicated to predicting Betweenness Centrality (BC) values.
4. **Execute 2-Hop Labeling**: We utilize the predicted BC values to sort vertices and execute the 2-Hop Labeling algorithm.

<div align=center><img alt="WORKFLOW"src="assets\pipeline.png"/></div>

## Repository Contents

### Code Structure

```sh
PathBee
├── README.md                  # Project README file
├── algorithms                 # Folder for algorithm-related code
│   ├── 2_hop_labeling.cpp     # 2-Hop Labeling algorithm implementation
│   ├── pruned_indexing.h      # Header file for pruned indexing algorithm
├── centralities               # Folder for centrality-related code
│   ├── cal_centrality.py      # Toolkit to calculate centrality values
│   ├── bc                     # Folder to store betweenness centrality calculation data
│   ├── close                  # Folder to store closeness centrality calculation data
│   ├── dc                     # Folder to store degree centrality calculation data
│   ├── eigen                  # Folder to store eigenvector centrality calculation data
│   ├── gnn                    # Folder for GNN-based centrality related code
│   └── gs                     # Folder to store GS betweenness centrality calculation data
├── gnn                        # Folder for GNN-related code
│   ├── models                 # Folder to store GNN model files
│   ├── predict                # Folder for GNN model inference code
│   │   ├── layer.py           # GNN layer implementation
│   │   ├── model_bet.py       # GNN model for betweenness centrality prediction
│   │   ├── predict.py         # GNN model inference script
│   │   ├── utils.py           # Utility functions for GNN model inference
│   │   └── utils_gnn.py       # Utility functions specific to GNNs
│   └── train                  # Folder for GNN model training code
│       ├── betweenness.py     # Script to train GNN for betweenness centrality
│       ├── layer.py           # GNN layer implementation
│       ├── model_bet.py       # GNN model for betweenness centrality training
│       └── utils.py           # Utility functions for GNN model training
├── graphs                     # Folder for graph-related code and data
│   ├── real_world             # Folder for real-world graph data
│   └── synthetic              # Folder for synthetic graph data
│       ├── create_dataset.py  # Script to create training and test datasets
│       ├── generate_graph.py  # Script to generate scale-free graphs
│       ├── data_splits        # Folder to store data splits for training and testing
│       └── graphs             # Folder to store generated graph files
└── requirements.txt           # List of required Python packages
```

### Key Files Description

`graphs/synthetic/generate_graph.py`: This script generates synthetic scale-free graph data, which serves as the foundational dataset for Graph Neural Network (GNN) training.

`gnn/train/betweenness.py`: This script is used to train GNN models, utilizing Betweenness Centrality as a feature. The aim is to create exceptional models for predicting Betweenness Centrality (BC).

`gnn/predict/predict.py`: This script employs the trained GNN model to predict BC values. The result serves as an ORDER, swiftly generating high-quality ORDERS, which are calculated at runtime, for the 2-hop-labeling algorithm.

`algorithms/2_hop_labeling.cpp`: A comprehensive implementation of the 2-hop-labeling algorithm. This can be expanded by integrating a new 2-hop-labeling algorithm into the existing framework.


## Quick Start

### Step 0: Prerequisites

Ensure you're running Python 3.9 and CUDA 11.1 or higher. Install the necessary packages with:

```sh
pip install -r requirements.txt
```

To execute the 2_hop_labeling, you need to have GCC 11.3.0 or above installed.

### Step 1: Training Data Preparation

```sh
# Generate scale-free graph using NetworkX
python graphs/synthetic/generate_graph.py

# Create training and test datasets
python graphs/synthetic/create_dataset.py
```

### Step 2: GNN Model Training

```sh
# Train the GNN model for betweenness centrality
python gnn/train/betweenness.py
```

### Step 3: GNN Model Inference

```sh
# Run GNN model inference to predict BC values
python gnn/predict/predict.py <graph_path> <model_path>

# Example command:
python gnn/predict/predict.py graphs/real_world/map.txt gnn/models/example.pt
```

### Step 4: Execute 2-Hop Labeling

```sh
# Compile the 2-Hop Labeling algorithm
g++ -fopenmp algorithms/2_hop_labeling.cpp -o 2_hop_labeling

# Run the 2-Hop Labeling algorithm with the specified parameters
./2_hop_labeling <graph_path> <centrality_path> <algorithm_type>

# Example command:
./2_hop_labeling graphs/real_world/map.txt centralities/bc/bc_test.txt

```

### Baseline Generation

To calculate various centrality values for the graph as a baseline for comparison, you can use the `cal_centrality_for_real_world.py` toolkit. The available centrality methods are:

```python
centrality_dict = {
    'dc': degree_centrality,  # Degree centrality
    'bc': betweenness_centrality,  # Betweenness centrality
    'gs': gs_betweenness_centrality,  # GS Betweenness centrality
    'kadabra': kadabra_betweenness_centrality,  # Kadabra Betweenness centrality
    'close': closeness_centrality,  # Closeness centrality
    'eigen': eigenvector_centrality  # Eigenvector centrality
}
```

To calculate centrality values using the baseline methods mentioned above, you can run the following command:

```sh
python centralities/cal_centrality.py -centrality <method>
```

Replace ``<method>`` with one of the available centrality methods from the centrality_dict above. This command will calculate the corresponding centrality values and provide them as a baseline for your research.

## Performance

### Baseline

PathBee is a general framework that can be used with various 2-hop labeling algorithms. The central component shared among these algorithms is the pruned index construction procedure. This repository demonstrates the effectiveness and versatility of PathBee by applying it to the representative offline 2-hop labeling algorithm, PLL [1], within the pruned index construction procedure. For in-depth insights into PathBee's integration with different algorithms, please refer to our paper for detailed information.

### Setup

For experiments in a single-core CPU environment, we use a Linux server with AMD Ryzen 7 3700X (3.6 GHz) and 128 GB of main memory. For experiments in a multi-core CPU environment, we use a Linux server with Intel(R) Xeon(R) CPU E5-2696 v4 (2.2 GHz, 80 cores) and 128 GB of main memory. The cut-off time is set to 12 hours.

### Effectiveness of PathBee on PLL

We conducted a comparison of the index construction time, index size and query time between PLL, PathBee, and PathBee+ on 26 real-world datasets.

The performance comparison result is shown as below.

#### Comparison on Index Construction Time

<div align=center><img alt="pll_IBT"src="assets\pll_IBT.png"/></div>

#### Comparison on Index Size

<div align=center><img alt="pll_IS"src="assets\pll_IS.png"/></div>

#### Comparison on Query Time

<div align=center><img alt="pll_IT"src="assets\pll_QT.png"/></div>

### Scalability of PathBee
In the experiment for scalability validation, we randomly divide vertices of a graph into 5 equally sized vertex groups and create 5 graphs for the cases of 20%, 40%, 60%, 80%, 100%: the 𝑖-th graph is the induced subgraph on the first 𝑖-th vertex group.

The scalability validation result is shown as below.

#### Scalability Tests

<div align=center><img alt="sca_IBT"src="assets\sca-PAT.png"/></div>

<div align=center><img alt="sca_IS"src="assets\sca-IMDB.png"/></div>

### Traversal Cost Model Validation

The extra traversal cost model validation result (section 4 in our paper) is shown as below.

<div align="center">
  <img alt="sca_IBT" src="assets\model-EPI.png" width="340"/>
  <img alt="sca_IBT" src="assets\model-MOC.png" width="340"/>
</div>

<div align="center">
  <img alt="sca_IBT" src="assets\model-GN.png" width="340"/>
  <img alt="sca_IBT" src="assets\model-SLA.png" width="347"/>
</div>

## References

[1] [Pruned Landmark Labeling](https://github.com/iwiwi/pruned-landmark-labeling)

[2] [GNN-Bet](https://github.com/sunilkmaurya/GNN-Bet)
