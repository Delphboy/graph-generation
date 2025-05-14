import torch
from torch_geometric.data import HeteroData

from generators import generator

class Attention(generator.Generator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def build_graph(self, idx: int):
        data = HeteroData()

        # objects
        data["object"].x = self._get_objects(idx)
        num_objects = data["object"].x.shape[0]

        node_indices = torch.arange(num_objects)
        src, dst = torch.meshgrid(node_indices, node_indices, indexing="ij")
        complete_edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
        assert complete_edge_index.shape[1] == num_objects + num_objects * (num_objects - 1), f"ATTENTION - BAD OBJECT-ATTENTION-OBJECT: Expected {num_objects + num_objects * (num_objects - 1)} Received {complete_edge_index.shape[1]}"
        data["object", "attention", "object"].edge_index = complete_edge_index

        return data

