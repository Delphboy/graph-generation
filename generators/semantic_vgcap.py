import numpy as np

import torch
from torch_geometric.data import HeteroData

from generators import generator

class SemanticVgCap(generator.Generator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def build_graph(self, idx: int):
        data = HeteroData()
        nodes, edges = self._get_vgcap_data(idx)
        objects = self._get_objects(idx)

        assert nodes.shape[0] == objects.shape[0], f"SHAPE MISMATCH: Unable to link nodes ({nodes.shape}) and objects ({objects.shape}) in {idx}. "

        edge_index = edges[:, :2]
        edge_attr = edges[:, 2]

        data["object"].x = objects
        data["object", "semantic_relationships", "object"].edge_index = torch.from_numpy(edge_index.T)
        data["object", "semantic_relationships", "object"].edge_attr = torch.from_numpy(edge_attr)

        return data

