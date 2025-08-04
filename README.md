# PathBee: A Generic, Plug-and-Play Optimization Framework for Distance Labeling Algorithms (SIGMOD 2026)

This repository hosts the source code and supplementary materials for our SIGMOD 2026 paper, "A Generic, Plug-and-Play Optimization Framework for Distance Labeling Algorithms". This work presents PathBee, an innovative framework leveraging Graph Neural Networks (GNNs) that significantly advances the current 2-hop labeling-based approaches.

## PathBee Workflow

Our workflow is comprised of four key stages:

1. **Training Data Preparation**: We generate synthetic scale-free graphs using the NetworkX library, and then annotate the data using an internal tool that calculates precise centrality values.
2. **GNN Model Training**: We use the prepared datasets to train our GNN model.
3. **GNN Model Inference**: This stage is dedicated to predicting Betweenness Centrality (BC) values.
4. **Execute 2-Hop Labeling**: We utilize the predicted BC values to sort nodes and execute the 2-Hop Labeling algorithm.

![Workflow](assets/pipeline.png)

## Code Structure

```sh
PathBee
├── 2_hop_labeling              # Folder for 2-hop labeling algorithm
├── datasets                    # Folder for data-related files
├── launch.py                   # Script to launch the application
├── models                      # Folder for storing model files
├── pathbee                     # Main folder for core algorithms and functions
│   ├── algorithms              # Folder for algorithm-related code
│   └── gnn                     # Folder for GNN-related code
```

## Quick Start

### Step 0: Prerequisites

Ensure you're running Python 3.10 or higher. Install the necessary packages with:

```sh
python create -n pb python=3.10

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

To execute the 2_hop_labeling, you need to have GCC 11.3.0 or above installed.

### Step 1: Training Data Preparation

```sh
# Create training and test datasets
python launch.py gen 
```

### Step 2: GNN Model Training

```sh
# Train the GNN model for betweenness centrality
python launch.py train
```

### Step 3: GNN Model Inference

```sh
# Run GNN model inference to predict BC values
python launch.py cen --graph-path <graph_file> --centrality gnn_pb --model-path <model_path>
```

### Step 4: Execute 2-Hop Labeling

```sh
# Compile the 2-Hop Labeling algorithm
python launch.py index --graph-path <graph_file> --centrality-path <centrality_file> --index-path <index_file>
```

## Performance

Below we show our main experimental results comparing PathBee, PathBee+, and PLL on 26 real-world datasets. More results and details can be found in our paper.

#### Comparison on Index Construction Time

<div align=center><img alt="pll_IBT"src="assets\pll_IBT.png"/></div>

#### Comparison on Index Size

<div align=center><img alt="pll_IS"src="assets\pll_IS.png"/></div>

#### Comparison on Query Time

<div align=center><img alt="pll_IT"src="assets\pll_QT.png"/></div>

## NP-hard Proof
The proof of NP-hardness can be found in [assets/PathBee_NP_Hard_Proof.pdf](assets/PathBee_NP_Hard_Proof.pdf).

## References

[1] [Pruned Landmark Labeling](https://github.com/iwiwi/pruned-landmark-labeling)
[2] [Historical Pruned Landmark Labeling](https://github.com/iwiwi/historical-pruned-landmark-labeling)
[3] [GNN-Bet](https://github.com/sunilkmaurya/GNN-Bet)

