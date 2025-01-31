# PathBee: Accelerating Shortest Path Querying via Graph Neural Networks (VLDB 2025)

This repository hosts the source code and supplementary materials for our VLDB 2025 paper, "PathBee: Accelerating Shortest Path Querying via Graph Neural Networks". This work presents PathBee, an innovative framework leveraging Graph Neural Networks (GNNs) that significantly advances the current 2-hop labeling-based approaches.

## PathBee Workflow

Our workflow is comprised of four key stages:

1. **Training Data Preparation**: We generate synthetic scale-free graphs using the NetworkX library, and then annotate the data using an internal tool that calculates precise centrality values.
2. **GNN Model Training**: We use the prepared datasets to train our GNN model.
3. **GNN Model Inference**: This stage is dedicated to predicting Betweenness Centrality (BC) values.
4. **Execute 2-Hop Labeling**: We utilize the predicted BC values to sort nodes and execute the 2-Hop Labeling algorithm.

![Workflow](assets\pipeline.png)

## Code Structure

_(Provide a brief description of the main components of your codebase here.)_

## Quick Start

### Step 0: Prerequisites

Ensure you're running Python 3.9 and CUDA 11.1 or higher. Install the necessary packages with:

```sh
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
python launch.py infer
```

### Step 4: Execute 2-Hop Labeling

```sh
# Compile the 2-Hop Labeling algorithm
python launch.py pll
```

<!-- ## Baseline

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
python centralities/cal_centrality_for_real_world.py -centrality <method>
```

Replace ``<method>`` with one of the available centrality methods from the centrality_dict above. This command will calculate the corresponding centrality values and provide them as a baseline for your research. -->

## References

[1] [Pruned Landmark Labeling](https://github.com/iwiwi/pruned-landmark-labeling)
[2] [Historical Pruned Landmark Labeling](https://github.com/iwiwi/historical-pruned-landmark-labeling)
[3] [GNN-Bet](https://github.com/sunilkmaurya/GNN-Bet)
