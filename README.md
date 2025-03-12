# COCO Heterogeneous Graph Generation

Precompute PyG `HeteroData` objects from [BUTD](https://arxiv.org/abs/1707.07998), [VSUA](https://arxiv.org/abs/1908.02127), and [SGAE](https://arxiv.org/abs/1812.02378) data.

## Setup

Please see [this guide](https://henrysenior.com/words/2024-04-03-coco-supplementary-dataset-download-guide) for instructions on how to download, extract and setup all the required data. You may also find [this explanation](https://henrysenior.com/words/2024-05-06coco-semantic-graph-data) of the data helpful.

The python environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python3 -m pip install torch_geometric
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

## Graph Types

Below is a list of the graph types supported by this project. Please see Section 2.3 of our [survey paper](https://henrysenior.com/publications/graph-neural-networks-in-vision-language-image-understanding-a-survey) for a visual overview of the different graph types that can be used to represent a scene. 

The project currently supports the following graph types:
- **semantic**: Follows the [GCN-LSTM](https://arxiv.org/abs/1809.07041) definition of a semantic graph i.e. objects connected via edges with one-hot-encoded edge labels describing the semantic relationship between the two objects.
- **geometric**: Follows the [VSUA](https://arxiv.org/abs/1908.02127) 'geometric' graph i.e. objects connected via edges with an $\mathbb{R}^8$ feature describing the spatial relationship between the two objects.
- **heteroyao**: A heterogeneous graph that combines both the semantic and geometric/spatial edges, essentially merging the two graph types used by the [GCN-LSTM](https://arxiv.org/abs/1809.07041) paper by Yao el al. Note that this graph does not use any attribute features. 


### Adding New Graphs

To add in a new custom graph defintion, simply extend the `Generator` class and override the `build_graph` function. Then extend the `_GRAPH_TYPES` dictionary in `main.py`.

## Options

See the table below for an overview of the options and `run.sh` for an example of how they come together.

| Options | Information |
|--|--|
|`--butd` | The path to the directory containing the butd object feature data |
|`--vsua` | The path to the directory containing the vsua geometric edge data |
|`--sgae` | The path to the directory containing the sgae semantic relationhship data |
|`--output_dir` | The directory where the `HeteroData` objects will be saved | 
|`--graph_type` | The graph type to generate. See above for a list of options. Must be given in lowercase |



